#!/usr/bin/env python3
"""
Template-Based IDP Design with Your GFlowNet
=============================================

Integrates with your existing GFlowNet training code to enable
template-based generation with fixed motifs and variable regions.

Usage:
    python template_idp_design.py --model_path saved_model.pt --template "[PAD:20]RGGRGG[PAD:30]"
    
Author: Adapted for your GFlowNet
Date: 2025
"""

import torch
import numpy as np
import json
import argparse
import re
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
import pandas as pd
from collections import Counter

# Import your existing components
from gfnxidp import get_tokenizer, get_oracle, get_default_args
from gfnxidp.generator import TBGFlowNetGenerator
from gfnxidp.utils import X_from_seq, Model, AttrSetter


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class TemplateSegment:
    """Represents a segment in the template"""
    type: str  # 'fixed' or 'variable'
    content: Union[str, int]  # sequence (if fixed) or length (if variable)
    start_pos: Optional[int] = None
    end_pos: Optional[int] = None


@dataclass
class DesignResult:
    """Result from template-based generation"""
    sequence: str
    oracle_score: float
    template: str
    motif_positions: List[Tuple[int, str]]
    constraint_penalty: float
    properties: Dict[str, float]


# ============================================================================
# TEMPLATE PARSER
# ============================================================================

class TemplateParser:
    """Parse template strings into structured segments"""
    
    @staticmethod
    def parse_string_template(template_string: str) -> List[TemplateSegment]:
        """
        Parse template string into segments.
        
        Example: "[PAD:20]RGGRGG[PAD:30]FGFG[PAD:20]"
        
        Returns:
            List of TemplateSegment objects
        """
        segments = []
        pattern = r'\[PAD:(\d+)\]|([A-Z]+)'
        
        current_pos = 0
        for match in re.finditer(pattern, template_string):
            if match.group(1):  # Padding region
                length = int(match.group(1))
                segments.append(TemplateSegment(
                    type='variable',
                    content=length,
                    start_pos=current_pos,
                    end_pos=current_pos + length
                ))
                current_pos += length
                
            elif match.group(2):  # Fixed motif
                motif = match.group(2)
                segments.append(TemplateSegment(
                    type='fixed',
                    content=motif,
                    start_pos=current_pos,
                    end_pos=current_pos + len(motif)
                ))
                current_pos += len(motif)
        
        return segments
    
    @staticmethod
    def create_position_template(total_length: int,
                                motif_placements: List[Tuple[int, str]]) -> List[TemplateSegment]:
        """
        Create template from explicit motif positions.
        
        Args:
            total_length: Total sequence length
            motif_placements: List of (position, motif) tuples
        """
        sorted_placements = sorted(motif_placements, key=lambda x: x[0])
        
        segments = []
        current_pos = 0
        
        for pos, motif in sorted_placements:
            # Add padding before this motif
            if pos > current_pos:
                segments.append(TemplateSegment(
                    type='variable',
                    content=pos - current_pos,
                    start_pos=current_pos,
                    end_pos=pos
                ))
            
            # Add fixed motif
            segments.append(TemplateSegment(
                type='fixed',
                content=motif,
                start_pos=pos,
                end_pos=pos + len(motif)
            ))
            
            current_pos = pos + len(motif)
        
        # Add final padding if needed
        if current_pos < total_length:
            segments.append(TemplateSegment(
                type='variable',
                content=total_length - current_pos,
                start_pos=current_pos,
                end_pos=total_length
            ))
        
        return segments


# ============================================================================
# TEMPLATE-AWARE GFLOWNET GENERATOR
# ============================================================================

class TemplateGFlowNetGenerator:
    """
    Adapter for your GFlowNet that enables template-based generation.
    Generates variable regions while respecting fixed motifs.
    """
    
    def __init__(self, gflownet_model, tokenizer, args):
        """
        Args:
            gflownet_model: Your trained TBGFlowNetGenerator
            tokenizer: Your tokenizer
            args: Configuration arguments
        """
        self.model = gflownet_model
        self.tokenizer = tokenizer
        self.args = args
        self.device = args.device
        self.vocab = tokenizer.vocab  # Skip EOS and PAD tokens
        
        # Sampling parameters
        self.temperature = getattr(args, 'gen_sampling_temperature', 1.0)
        self.out_coef = getattr(args, 'gen_output_coef', 1.0)
    
    def generate_from_template(self,
                              segments: List[TemplateSegment],
                              n_samples: int = 1) -> List[List[int]]:
        """
        Generate sequences from template segments.
        
        Args:
            segments: List of TemplateSegment objects
            n_samples: Number of sequences to generate
            
        Returns:
            List of token sequences
        """
        all_sequences = []
        
        for _ in range(n_samples):
            # Build sequence step by step following template
            current_tokens = []
            
            for segment in segments:
                if segment.type == 'fixed':
                    # Add fixed motif tokens
                    motif = segment.content
                    motif_tokens = self.tokenizer.tokenize(motif)
                    current_tokens.extend(motif_tokens)
                
                elif segment.type == 'variable':
                    # Generate variable region using GFlowNet
                    length = segment.content
                    fragment_tokens = self._generate_fragment(
                        current_tokens, 
                        length
                    )
                    current_tokens.extend(fragment_tokens)
            
            all_sequences.append(current_tokens)
        
        return all_sequences
    
    def _generate_fragment(self, 
                          context_tokens: List[int],
                          length: int) -> List[int]:
        """
        Generate a fragment of specified length given context.
        
        Args:
            context_tokens: Preceding tokens (for context)
            length: Number of tokens to generate
            
        Returns:
            List of generated tokens
        """
        generated = []
        current_state = context_tokens.copy()
        
        for position in range(length):
            # Prepare input for model
            x = self.tokenizer.pad_tokens([current_state]).to(self.device)
            
            # Get logits from GFlowNet
            with torch.no_grad():
                logits = self.model(x, None, coef=self.out_coef)
            
            # Apply temperature sampling
            logits = logits / self.temperature
            
            # Prevent EOS token (we want to continue generating)
            logits[:, 0] = -1000
            
            # Sample next token
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            
            generated.append(next_token)
            current_state.append(next_token)
        
        return generated


# ============================================================================
# PROPERTY CALCULATOR (Uses your existing oracle)
# ============================================================================

class PropertyCalculator:
    """Calculate IDP properties using your existing infrastructure"""
    
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        
        # Amino acid groups
        self.hydrophobic_aa = set('AILMFWVY')
        self.aromatic_aa = set('FWY')
    
    def calculate_all_properties(self, sequence: str) -> Dict[str, float]:
        """Calculate biophysical properties"""
        if not sequence:
            return {
                'length': 0,
                'cys_fraction': 0.0,
                'hydrophobic_fraction': 0.0,
                'aromatic_fraction': 0.0,
                'charge': 0.0,
                'fcr': 0.0
            }
        
        length = len(sequence)
        
        # Cysteine content
        cys_frac = sequence.count('C') / length
        
        # Hydrophobic content
        hydro_count = sum(1 for aa in sequence if aa in self.hydrophobic_aa)
        hydro_frac = hydro_count / length
        
        # Aromatic content
        arom_count = sum(1 for aa in sequence if aa in self.aromatic_aa)
        arom_frac = arom_count / length
        
        # Charge properties
        pos_charged = sum(sequence.count(aa) for aa in 'KR')
        neg_charged = sum(sequence.count(aa) for aa in 'DE')
        net_charge = pos_charged - neg_charged
        fcr = (pos_charged + neg_charged) / length
        
        return {
            'length': length,
            'cys_fraction': cys_frac,
            'hydrophobic_fraction': hydro_frac,
            'aromatic_fraction': arom_frac,
            'charge': net_charge,
            'fcr': fcr,
            'positive_charge': pos_charged,
            'negative_charge': neg_charged
        }


# ============================================================================
# CONSTRAINT EVALUATOR (Uses your existing penalty system)
# ============================================================================

class ConstraintEvaluator:
    """Evaluate constraint violations using your existing penalty system"""
    
    def __init__(self, constraint_penalty):
        """
        Args:
            constraint_penalty: Your IDPConstraintPenalty instance
        """
        self.constraint_penalty = constraint_penalty
    
    def evaluate(self, sequence: str) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate constraints on a sequence.
        
        Returns:
            (total_penalty, breakdown) where penalty is in (0, 1]
        """
        if self.constraint_penalty is None:
            return 1.0, {}
        
        total, breakdown = self.constraint_penalty.compute_total_penalty(sequence)
        return total, breakdown


# ============================================================================
# MAIN TEMPLATE DESIGNER
# ============================================================================

class TemplateIDPDesigner:
    """
    Main class for template-based IDP design using your GFlowNet.
    """
    
    def __init__(self, args, gflownet_model, oracle, tokenizer, constraint_penalty=None):
        """
        Args:
            args: Configuration arguments
            gflownet_model: Your trained TBGFlowNetGenerator
            oracle: Your IDPOracle instance
            tokenizer: Your tokenizer
            constraint_penalty: Optional IDPConstraintPenalty
        """
        self.args = args
        self.oracle = oracle
        self.tokenizer = tokenizer
        
        # Initialize components
        self.generator = TemplateGFlowNetGenerator(gflownet_model, tokenizer, args)
        self.property_calc = PropertyCalculator(args, tokenizer)
        self.constraint_eval = ConstraintEvaluator(constraint_penalty)
    
    def design_from_template(self,
                           template: Union[str, List[TemplateSegment]],
                           n_candidates: int = 100,
                           verbose: bool = True) -> List[DesignResult]:
        """
        Generate IDP designs from template.
        
        Args:
            template: Template string or list of segments
            n_candidates: Number of candidates to generate
            verbose: Print progress
            
        Returns:
            List of DesignResult objects, sorted by oracle score
        """
        # Parse template if string
        if isinstance(template, str):
            segments = TemplateParser.parse_string_template(template)
            template_str = template
        else:
            segments = template
            template_str = self._segments_to_string(segments)
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Template-Based IDP Design")
            print(f"{'='*70}")
            print(f"Template: {template_str}")
            print(f"Total length: {sum(s.content if s.type == 'variable' else len(s.content) for s in segments)}")
            print(f"Generating {n_candidates} candidates...")
        
        # Generate candidates
        token_sequences = self.generator.generate_from_template(segments, n_samples=n_candidates)
        
        if verbose:
            print(f"Evaluating with oracle ({self.oracle.model_type})...")
        
        # Evaluate all candidates
        results = []
        for i, tokens in enumerate(token_sequences):
            if verbose and (i + 1) % 20 == 0:
                print(f"  Progress: {i + 1}/{n_candidates}")
            
            # Convert to sequence
            sequence = self.tokenizer.detokenize(tokens)
            
            # Get oracle score
            oracle_score = self.oracle.predict(tokens)
            
            # Calculate properties
            properties = self.property_calc.calculate_all_properties(sequence)
            
            # Evaluate constraints
            constraint_penalty, constraint_breakdown = self.constraint_eval.evaluate(sequence)
            
            # Find motif positions
            motif_positions = self._find_motif_positions(sequence, segments)
            
            # Create result
            result = DesignResult(
                sequence=sequence,
                oracle_score=float(oracle_score),
                template=template_str,
                motif_positions=motif_positions,
                constraint_penalty=float(constraint_penalty),
                properties=properties
            )
            
            results.append(result)
        
        # Sort by oracle score (higher is better for CSAT, lower for DG)
        results.sort(key=lambda x: x.oracle_score)
        
        if verbose:
            self._print_summary(results)
        
        return results
    
    def design_with_auto_placement(self,
                                  total_length: int,
                                  motifs: List[str],
                                  n_candidates: int = 100,
                                  min_spacing: int = 15,
                                  max_attempts: int = 10,
                                  verbose: bool = True) -> List[DesignResult]:
        """
        Generate designs with automatic motif placement.
        
        Args:
            total_length: Total sequence length
            motifs: List of motifs to include
            n_candidates: Number of candidates per placement
            min_spacing: Minimum spacing between motifs
            max_attempts: Number of different placements to try
            verbose: Print progress
            
        Returns:
            List of DesignResult objects from all placements
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"Auto-Placement IDP Design")
            print(f"{'='*70}")
            print(f"Total length: {total_length}")
            print(f"Motifs: {motifs}")
            print(f"Trying {max_attempts} different placements...")
        
        all_results = []
        
        for attempt in range(max_attempts):
            if verbose:
                print(f"\n--- Placement {attempt + 1}/{max_attempts} ---")
            
            # Generate random positions
            motif_positions = self._sample_motif_positions(
                motifs, total_length, min_spacing
            )
            
            if verbose:
                print(f"Positions: {motif_positions}")
            
            # Create template
            segments = TemplateParser.create_position_template(
                total_length, motif_positions
            )
            
            # Generate candidates for this placement
            results = self.design_from_template(
                segments, 
                n_candidates=n_candidates,
                verbose=False
            )
            
            all_results.extend(results)
            
            if verbose:
                best = results[0]
                print(f"Best score: {best.oracle_score:.4f}, Constraint: {best.constraint_penalty:.3f}")
        
        # Sort all results
        reverse = (self.oracle.model_type == 'csat')
        all_results.sort(key=lambda x: x.oracle_score, reverse=reverse)
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Overall Best Results (from all placements)")
            print(f"{'='*70}")
            self._print_summary(all_results[:20])
        
        return all_results
    
    def _sample_motif_positions(self,
                               motifs: List[str],
                               total_length: int,
                               min_spacing: int) -> List[Tuple[int, str]]:
        """Sample valid positions for motifs"""
        positions = []
        
        # Calculate space needed
        total_motif_length = sum(len(m) for m in motifs)
        available_space = total_length - total_motif_length
        
        if available_space < min_spacing * (len(motifs) + 1):
            min_spacing = max(5, available_space // (len(motifs) + 1))
        
        current_pos = np.random.randint(min_spacing, max(min_spacing + 1, 2 * min_spacing))
        
        for motif in motifs:
            positions.append((current_pos, motif))
            current_pos += len(motif) + np.random.randint(min_spacing, max(min_spacing + 1, 2 * min_spacing))
        
        return positions
    
    def _find_motif_positions(self, 
                            sequence: str, 
                            segments: List[TemplateSegment]) -> List[Tuple[int, str]]:
        """Find positions of fixed motifs in the sequence"""
        positions = []
        current_pos = 0
        
        for segment in segments:
            if segment.type == 'fixed':
                motif = segment.content
                positions.append((current_pos, motif))
                current_pos += len(motif)
            else:
                current_pos += segment.content
        
        return positions
    
    def _segments_to_string(self, segments: List[TemplateSegment]) -> str:
        """Convert segments to readable template string"""
        parts = []
        for seg in segments:
            if seg.type == 'fixed':
                parts.append(seg.content)
            else:
                parts.append(f"[PAD:{seg.content}]")
        return ''.join(parts)
    
    def _print_summary(self, results: List[DesignResult]):
        """Print summary of results"""
        print(f"\n{'='*70}")
        print(f"Results Summary")
        print(f"{'='*70}")
        
        scores = [r.oracle_score for r in results]
        constraints = [r.constraint_penalty for r in results]
        
        print(f"\nOracle scores ({self.oracle.model_type}):")
        print(f"  Mean: {np.mean(scores):.4f}")
        print(f"  Std:  {np.std(scores):.4f}")
        print(f"  Best: {scores[0]:.4f}")
        print(f"  Median: {np.median(scores):.4f}")
        
        print(f"\nConstraint penalties:")
        print(f"  Mean: {np.mean(constraints):.4f}")
        print(f"  Perfect (1.0): {sum(1 for c in constraints if c >= 0.99)}/{len(constraints)}")
        
        print(f"\nTop 5 designs:")
        for i, result in enumerate(results[:5], 1):
            print(f"\n{i}. Score: {result.oracle_score:.4f}, Constraint: {result.constraint_penalty:.3f}")
            print(f"   {result.sequence}")
            print(f"   Motifs: {result.motif_positions}")


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Template-based IDP design with your GFlowNet'
    )
    
    # Model loading
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to saved GFlowNet model checkpoint')
    parser.add_argument('--config', type=str,
                       help='Optional: Path to design configuration JSON')
    
    # Template specification (use one of these)
    parser.add_argument('--template', type=str,
                       help='Template string, e.g., "[PAD:20]RGGRGG[PAD:30]"')
    parser.add_argument('--motifs', type=str, nargs='+',
                       help='Motifs for auto-placement, e.g., RGGRGG FGFG')
    parser.add_argument('--total_length', type=int, default=256,
                       help='Total sequence length (for auto-placement)')
    parser.add_argument('--min_spacing', type=int, default=15,
                       help='Minimum spacing between motifs (for auto-placement)')
    
    # Generation parameters
    parser.add_argument('--n_candidates', type=int, default=100,
                       help='Number of candidates to generate')
    parser.add_argument('--temperature', type=float, default=3.0,
                       help='Sampling temperature')
    parser.add_argument('--max_attempts', type=int, default=10,
                       help='Number of placement attempts (auto-placement only)')
    
    # Output
    parser.add_argument('--output', type=str, default='template_results.csv',
                       help='Output CSV file')
    
    args_cli = parser.parse_args()
    
    # Load default args and override
    args = get_default_args()
    args.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    args.gen_sampling_temperature = args_cli.temperature
    
    print(f"\n{'='*70}")
    print(f"Template-Based IDP Design")
    print(f"{'='*70}")
    
    # Initialize components
    print("\n1. Loading tokenizer and oracle...")
    tokenizer = get_tokenizer(args)
    oracle = get_oracle(args, tokenizer)
    print(f"   Oracle mode: {oracle.model_type}")
    
    # Load trained model
    print(f"\n2. Loading GFlowNet model from {args_cli.model_path}...")
    from gfnxidp import get_generator
    generator = get_generator(args, tokenizer)
    
    checkpoint = torch.load(args_cli.model_path, map_location=args.device)
    generator.model.load_state_dict(checkpoint['model_state_dict'])
    generator.model.eval()
    print(f"   Model loaded successfully")
    
    # Initialize constraint penalty
    print("\n3. Setting up constraints...")
    from gfnxidp.proxy import IDPConstraintPenalty
    constraint_penalty = IDPConstraintPenalty(args, tokenizer)
    
    # Create designer
    designer = TemplateIDPDesigner(args, generator, oracle, tokenizer, constraint_penalty)
    
    # Determine design mode
    if args_cli.template:
        print(f"\n4. Using template mode...")
        results = designer.design_from_template(
            template=args_cli.template,
            n_candidates=args_cli.n_candidates,
            verbose=True
        )
    elif args_cli.motifs:
        print(f"\n4. Using auto-placement mode...")
        results = designer.design_with_auto_placement(
            total_length=args_cli.total_length,
            motifs=args_cli.motifs,
            n_candidates=args_cli.n_candidates,
            min_spacing=args_cli.min_spacing,
            max_attempts=args_cli.max_attempts,
            verbose=True
        )
    else:
        raise ValueError("Must specify either --template or --motifs")
    
    # Save results
    print(f"\n5. Saving results to {args_cli.output}...")
    results_data = []
    
    for i, result in enumerate(results):
        row = {
            'rank': i + 1,
            'sequence': result.sequence,
            'oracle_score': result.oracle_score,
            'constraint_penalty': result.constraint_penalty,
            'template': result.template,
            'motif_positions': str(result.motif_positions),
        }
        row.update({f'prop_{k}': v for k, v in result.properties.items()})
        results_data.append(row)
    
    df = pd.DataFrame(results_data)
    df.to_csv(args_cli.output, index=False)
    
    print(f"\n✓ Results saved to {args_cli.output}")
    print(f"{'='*70}\n")


# ============================================================================
# PROGRAMMATIC EXAMPLE
# ============================================================================

def example_usage():
    """Example showing programmatic usage"""
    
    # Setup
    args = get_default_args()
    args.device = torch.device('cpu')
    
    tokenizer = get_tokenizer(args)
    oracle = get_oracle(args, tokenizer)
    
    # Load your trained model
    from gfnxidp import get_generator
    generator = get_generator(args, tokenizer)
    
    # Load checkpoint if you have one
    # checkpoint = torch.load('saved_model.pt')
    # generator.model.load_state_dict(checkpoint['model_state_dict'])
    
    from gfnxidp.proxy import IDPConstraintPenalty
    constraint_penalty = IDPConstraintPenalty(args, tokenizer)
    
    # Create designer
    designer = TemplateIDPDesigner(args, generator, oracle, tokenizer, constraint_penalty)
    
    # Example 1: String template
    print("\n=== Example 1: Fixed Template ===")
    results = designer.design_from_template(
        template="[PAD:20]RGGRGG[PAD:30]FGFG[PAD:20]",
        n_candidates=10
    )
    print(f"Best sequence: {results[0].sequence}")
    print(f"Oracle score: {results[0].oracle_score:.4f}")
    
    # Example 2: Auto-placement
    print("\n=== Example 2: Auto-Placement ===")
    results = designer.design_with_auto_placement(
        total_length=150,
        motifs=['RGGRGG', 'FGFG', 'SYGQ'],
        n_candidates=20,
        min_spacing=15,
        max_attempts=5
    )
    print(f"Best sequence: {results[0].sequence}")
    print(f"Motif positions: {results[0].motif_positions}")


if __name__ == '__main__':
    # For CLI usage:
    main()
    
    # For programmatic example (comment out main() above):
    # example_usage()