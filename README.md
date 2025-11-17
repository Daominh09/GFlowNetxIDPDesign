# 🧬 GFN-xIDP — GFlowNet-Based Intrinsically Disordered Protein Design

GFN-xIDP is a modular framework for **intrinsically disordered protein (IDP) design** using GFlowNets, proxy predictors, and biophysical oracles.

The project integrates:
- A GFlowNet sequence generator
- Physics-based and embedding-based oracle models
- Custom tokenizers and datasets
- IUPred2A disorder scoring
- HPC/Slurm execution
- Reproducible Conda-based environments
- Pluggable "tools" such as pretrained models, embeddings, and residue properties

The framework is designed for **research**, **protein engineering**, and **computational biology experiments** involving sequence optimization under biophysical constraints.

---

## 📁 Project Structure
```
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
│           ├── iupred2a_lib.py
│           └── iupred2a.py
├── environment.yml           # Conda environment specification
├── pyproject.toml            # Package metadata + dev setup
└── README.md                 # This file
```

---

## 🛠 Installation & Setup

### 1. Create the Conda environment
```bash
conda env create -f environment.yml
conda activate GFNxIDP
```

### 2. Update the environment file after adding packages
```bash
conda env export --from-history > environment.yml
```

### 3. Install the package in editable mode
```bash
python -m pip install -e .
```

---

## 🚀 Running Experiments

Run the main experiment script:
```bash
python scripts/run_idp.py
```

---

## 📚 Citation

If you use GFN-xIDP in your research, please cite:
```bibtex
@software{gfn_xidp,
  title = {GFN-xIDP: GFlowNet-Based Intrinsically Disordered Protein Design},
  author = {Tuan Minh Dao},
  year = {2025},
  url = {https://github.com/Daominh09/GFlowNetxIDPDesign}
}
```

---

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.

---


## 📧 Contact

For questions or collaboration inquiries, please contact [minhdao.work.616@gmail.com](mailto:minhdao.work.616@gmail.com).