🧬 GFN-xIDP — GFlowNet-Based Intrinsically Disordered Protein Design
GFN-xIDP is a modular framework for intrinsically disordered protein (IDP) design using GFlowNets, proxy predictors, and biophysical oracles.
The project integrates:
a GFlowNet sequence generator
physics-based and embedding-based oracle models
custom tokenizers and datasets
IUPred2A disorder scoring
HPC/Slurm execution
reproducible Conda-based environments
pluggable “tools” such as pretrained models, embeddings, and residue properties
The framework is designed for research, protein engineering, and computational biology experiments involving sequence optimization under biophysical constraints.

📁 Project Structure
GFLOWNETXIDPDESIGN/
├── datasets/                 # Raw and processed datasets
├── logs/                     # Model logs & outputs
├── scripts/                  # Slurm scripts and entrypoints
│   └── run_idp.py
├── src/
│   └── gfnxidp/
│       ├── __init__.py
│       ├── args.py           # Central configuration (default Args)
│       ├── dataset.py        # Dataset loader + preprocessing
│       ├── generator.py      # GFlowNet generator
│       ├── oracle.py         # Biophysical + ML oracles
│       ├── proxy.py          # Proxy model 
│       ├── tokenizer.py      # Amino-acid tokenizer
│       ├── utils.py          # Misc utilities
│       └── tools/            # External models & helper scripts
│           ├── data/
│           ├── Embeddings/
│           ├── Models/
│           ├── iupred2a.py
│           └── iupred2a_lib.py
├── environment.yml           # Conda environment specification
├── pyproject.toml            # Package metadata + dev setup
└── README.md                 

🛠️ Installation & Setup
1. Create the Conda Environment
conda env create -f environment.yml
conda activate GFNxIDP
To update the file after installing new packages:
conda env export --from-history > environment.yml

2. Install the Package in Editable Mode
From the project root:
pip install -e .

🚀 Running Experiments
1. Running Locally
From the project root:
python scripts/run_idp.py
You can override defaults using CLI arguments:
python scripts/run_idp.py --gen_learning_rate 3e-4 --num_rounds 20
