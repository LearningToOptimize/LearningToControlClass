# Learning to Control Class - Course Repository

Special Topics on Optimal Control and Learning — Fall 2025 (ISYE 8803 VAN) at Georgia Institute of Technology. This is a Julia-based educational repository containing course materials delivered through interactive Pluto notebooks covering optimal control, numerical optimization, reinforcement learning, and PDE-constrained optimization.

**ALWAYS reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.**

## Working Effectively

### Repository Setup and Environment
- **Clone and navigate**: `git clone <repo-url> && cd LearningToControlClass`
- **Julia requirement**: Julia 1.10+ required. Check with `julia --version`
- **Install Julia**: Download from https://julialang.org/downloads (use current stable release for your OS)

### Documentation Build
- **CRITICAL**: Documentation builds quickly but package installs can be slow
- **Setup symlink for docs** (required first time):
  ```bash
  ln -s ../../class01 docs/src/class01
  ```
- **Build docs**: 
  ```bash
  cd docs/
  julia --project=. -e 'using Pkg; Pkg.instantiate()'  # Takes 60-90 seconds. NEVER CANCEL.
  cd ..  # Return to repo root
  julia --project=docs/ docs/make.jl  # Takes ~10 seconds
  ```
- **TIMEOUT WARNING**: Use 120+ second timeout for `Pkg.instantiate()` - network issues can cause delays
- **Documentation output**: Generated in `docs/build/` directory

### Working with Class Materials
- **Each class has its own Julia environment** in `classXX/` folders with `Project.toml` and `Manifest.toml` files
- **Install class dependencies**: 
  ```bash
  cd class01/  # or any other classXX folder
  julia --project=. -e 'using Pkg; Pkg.instantiate()'
  ```
- **CRITICAL TIMING**: Package installation takes 15-30 minutes on first run. NEVER CANCEL. Set timeout to 45+ minutes.
- **Network limitations**: If package downloads fail due to network issues, document the limitation but continue with available functionality

### Running Pluto Notebooks (Interactive Learning Environment)
- **Start Pluto server**:
  ```bash
  cd class01/  # or target class folder
  julia --project=.
  julia> using Pluto
  julia> Pluto.run()
  ```
- **Pluto startup time**: ~30-60 seconds. NEVER CANCEL.
- **Access**: Pluto opens web interface typically at http://localhost:1234
- **Open existing notebooks**: Enter full path to `.jl` file (e.g., `background_materials/math_basics.jl`) in Pluto interface
- **Available notebooks per class**: `*.jl` files in `background_materials/` and class root directories

### Repository Structure Navigation
```
/
├── .github/workflows/documentation.yml  # CI for docs
├── docs/                               # Documentation generation
│   ├── Project.toml                   # Docs dependencies
│   └── make.jl                        # Build script
├── classXX/                           # Individual class materials
│   ├── Project.toml                   # Class-specific dependencies
│   ├── Manifest.toml                  # Locked dependency versions
│   ├── classXX.md                     # Class overview and instructions
│   ├── background_materials/          # Pluto notebooks (.jl files)
│   └── *.html                         # Pre-generated notebook exports
└── README.md                          # Course overview
```

## Validation and Testing

### Manual Validation Scenarios
**ALWAYS test these workflows after making changes:**

1. **Basic Julia functionality**:
   ```bash
   julia -e 'println("Julia version: ", VERSION); using LinearAlgebra; println("Basic packages work!")'
   ```

2. **Documentation build validation**:
   ```bash
   # Ensure symlink exists first:
   ln -s ../../class01 docs/src/class01  # Only needed once
   julia --project=docs/ docs/make.jl
   # Should complete in ~10 seconds without errors
   # Verify docs/build/ directory contains generated HTML
   ```

4. **Class environment validation** (may fail due to network limitations):
   ```bash
   cd class01/
   julia --project=. -e 'using Pkg; Pkg.status()'
   # Should show all required packages listed (some may not be installed due to network issues)
   # Package loading may fail with "Artifact not found" errors - this is expected
   ```

4. **Pluto notebook basic test** (may fail due to network limitations):
   ```bash
   cd class01/
   julia --project=. -e 'using Pluto; println("Pluto loaded successfully!")'
   # May fail with "Artifact not found" errors due to network restrictions
   # Document the limitation but continue with available functionality
   ```

### Common Validation Commands
- **Check Julia installation**: `julia --version`
- **Package status**: `julia --project=. -e 'using Pkg; Pkg.status()'`
- **Precompile packages**: `julia --project=. -e 'using Pkg; Pkg.precompile()'`
- **Documentation links**: Verify generated links in `docs/build/index.html`

## Development Workflow

### Making Changes to Course Materials
- **Edit Pluto notebooks**: Modify `.jl` files directly or through Pluto interface
- **Update documentation**: Modify markdown files in class directories
- **Regenerate docs**: Run `julia --project=docs/ docs/make.jl` after changes
- **No traditional tests**: This is educational content - validation is through successful notebook execution

### Git Workflow Notes
- **Standard Git commands**: `git add`, `git commit`, `git push` work normally
- **GitHub Actions**: Documentation automatically builds on push to main branch
- **Class materials**: Each class is self-contained with its own dependencies

## Common Issues and Troubleshooting

### Expected Network Limitations
- **Package downloads frequently fail**: Due to firewall/network restrictions accessing pkg.julialang.org and GitHub binary artifacts
- **"Artifact not found" errors**: Common for packages requiring binary dependencies (OpenSSL, graphics libraries, etc.)
- **Pluto may not fully load**: Depends on HTTP package which requires network artifacts
- **Document these limitations**: Note which specific packages fail but continue with available functionality

### Package Installation Issues
- **Network timeouts**: Common due to downloading large binary dependencies
- **Manifest version warnings**: Normal when using different Julia versions than original (1.10.5 vs 1.11.6)
- **Failed downloads**: Document which packages fail but continue with available functionality

### Performance Expectations
- **Documentation build**: ~10 seconds - NEVER CANCEL
- **Package installation**: 15-30 minutes first time - NEVER CANCEL, timeout 45+ minutes  
- **Pluto startup**: 30-60 seconds - NEVER CANCEL (when it works)
- **Julia REPL startup**: 5-10 seconds

### Repository Limitations
- **No comprehensive test suite**: Educational repository focused on interactive learning
- **No linting/formatting tools**: Julia ecosystem tools available but not configured
- **Network dependencies**: Some functionality requires internet access for package downloads
- **Interactive UI testing**: Cannot fully test Pluto web interface in automated environments

## Key Learning Workflows

### File Types and Purposes
- **`.jl` files**: Pluto notebook source files (can be read as text, edited directly)
- **`.html` files**: Pre-generated static exports of notebooks (viewable in browser)
- **`.md` files**: Course documentation and instructions
- **`Project.toml`**: Julia package requirements for each class
- **`Manifest.toml`**: Exact package versions (locked dependencies)

### For Students/Users
1. **Setup**: Install Julia → Clone repo → Navigate to class folder → Install dependencies
2. **Learning**: Start Pluto → Open notebook → Work through interactive exercises
3. **Exploration**: Try different classes, modify examples, experiment with code
4. **Fallback**: If Pluto fails, read `.jl` files directly or view `.html` exports

### For Contributors/Developers
1. **Content changes**: Edit `.jl` notebooks or `.md` documentation  
2. **Build verification**: Regenerate docs to ensure no broken links
3. **Cross-class testing**: Verify changes don't break other class environments
4. **Documentation updates**: Keep class instructions synchronized with changes

**REMEMBER**: This is primarily an educational repository. Success is measured by students being able to run interactive notebooks and learn optimal control concepts, not by traditional software metrics like test coverage or build speed.