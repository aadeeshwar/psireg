# PSI Controller Research Suite

This directory contains a complete research suite for evaluating PSI controller performance against baseline controllers. Perfect for thesis work and research documentation.

## 📋 Overview

The PSI Controller Research Suite provides:
- **Complete simulation pipeline** for PSI controller evaluation
- **Comprehensive statistical analysis** with significance testing
- **Professional visualizations** for research presentations
- **Streamlined workflows** for reproducible research

## 🚀 Quick Start

### Option 1: Complete Study (Recommended)
Run everything in one command:
```bash
python scripts/run_complete_study.py
```

### Option 2: Individual Components
Run components individually:
```bash
# 1. Run PSI evaluation
python run_psi_evaluation.py

# 2. Quick summary
python scripts/quick_summary.py output/psi_evaluation_*.parquet

# 3. Comprehensive analysis
python scripts/analyze_results.py output/psi_evaluation_*.parquet

# 4. Full visualizations
python scripts/visualize_psi_results.py output/psi_evaluation_*.parquet
```

## 📁 Script Documentation

### 1. `run_complete_study.py` - Master Research Script
**Purpose**: Complete end-to-end research study
**What it does**:
- Runs PSI controller simulations across all scenarios
- Performs comprehensive statistical analysis
- Generates publication-quality visualizations
- Creates master research report
- Organizes all outputs in timestamped study directory

**Usage**:
```bash
python scripts/run_complete_study.py [--output-dir study_results]
```

**Outputs**:
- `study_results/psi_study_YYYYMMDD_HHMMSS/` - Complete study package
- Raw simulation data, analysis results, visualizations, and reports

### 2. `quick_summary.py` - Immediate Results Overview
**Purpose**: Quick overview of PSI controller performance
**What it does**:
- Loads simulation results
- Creates summary dashboard with key metrics
- Prints performance statistics to console
- Generates publication-ready summary chart

**Usage**:
```bash
python scripts/quick_summary.py <data_file.parquet>
```

**Outputs**:
- `psi_summary_dashboard.png` - 6-panel summary dashboard
- Console output with key statistics

### 3. `analyze_results.py` - Statistical Analysis
**Purpose**: Rigorous statistical analysis of results
**What it does**:
- Calculates comprehensive summary statistics
- Performs significance testing (t-tests, Mann-Whitney U)
- Computes confidence intervals
- Analyzes scenario-specific performance
- Generates performance rankings
- Creates detailed analysis report

**Usage**:
```bash
python scripts/analyze_results.py <data_file.parquet> [--output-dir analysis]
```

**Outputs**:
- `summary_statistics.csv` - Detailed descriptive statistics
- `significance_tests.csv` - Statistical significance results
- `confidence_intervals.csv` - 95% confidence intervals
- `scenario_analysis.csv` - Scenario-specific performance
- `performance_ranking.csv` - Overall controller ranking
- `analysis_report.md` - Comprehensive analysis report

### 4. `visualize_psi_results.py` - Comprehensive Visualizations
**Purpose**: Publication-quality interactive visualizations
**What it does**:
- Creates interactive Plotly charts
- Generates publication-ready matplotlib figures
- Produces multiple visualization types
- Saves both interactive (HTML) and static (PNG) formats

**Usage**:
```bash
python scripts/visualize_psi_results.py <data_file.parquet> [--output-dir visualizations]
```

**Outputs**:
- `performance_comparison.html` - Interactive performance comparison
- `scenario_analysis.html` - Scenario-specific analysis
- `time_series_analysis.html` - Time series charts
- `improvement_heatmap.html` - PSI improvement heatmap
- `radar_comparison.html` - Multi-metric radar chart
- `statistical_summary.html` - Statistical distributions
- `psi_summary_matplotlib.png` - Publication-ready summary

## 📊 Key Metrics Analyzed

### Primary Performance Indicators
- **Fossil Backup Energy (MWh)**: Primary efficiency metric
- **Grid Stability Score**: Frequency and voltage control performance
- **Renewable Utilization (%)**: Percentage of available renewable energy used
- **Curtailment (MW)**: Amount of renewable energy wasted
- **Frequency Deviation (Hz)**: Grid frequency stability
- **Voltage Deviation (V)**: Grid voltage stability

### Statistical Tests
- **T-tests**: Parametric significance testing
- **Mann-Whitney U tests**: Non-parametric significance testing
- **Cohen's d**: Effect size measurement
- **95% Confidence Intervals**: Uncertainty quantification

### Scenarios Tested
- **Routine Day**: Normal grid operations
- **Storm Day**: Severe weather conditions
- **Demand Spike**: High energy demand periods

## 🏆 Expected Results

Based on the PSI controller's design, you should expect:

### Performance Improvements
- **50-75% reduction** in fossil fuel backup vs Rule-based controller
- **40-60% reduction** in fossil fuel backup vs ML-only controller
- **30-50% reduction** in fossil fuel backup vs Swarm-only controller

### Grid Stability
- **Superior frequency control** (lower deviation from 60 Hz)
- **Enhanced voltage stability** (lower deviation from 230 V)
- **Improved overall grid stability scores**

### Renewable Integration
- **Higher renewable utilization rates**
- **Lower curtailment** of clean energy
- **Better predictive optimization**

## 📈 Visualization Types

### Interactive Charts (HTML)
1. **Performance Comparison**: Multi-metric controller comparison
2. **Scenario Analysis**: Performance across different scenarios
3. **Time Series**: Hourly performance patterns
4. **Improvement Heatmap**: PSI improvements vs baselines
5. **Radar Chart**: Multi-dimensional performance visualization
6. **Statistical Summary**: Distribution analysis with box plots

### Static Charts (PNG)
1. **Summary Dashboard**: 6-panel overview for presentations
2. **Publication Figure**: Comprehensive matplotlib summary

## 🔄 Workflow for Thesis Research

### Step 1: Generate Complete Study
```bash
python scripts/run_complete_study.py
```

### Step 2: Review Results
1. Open `study_results/psi_study_*/reports/PSI_Controller_Research_Report.md`
2. Review interactive visualizations in `visualizations/` directory
3. Examine statistical analysis in `analysis/` directory

### Step 3: Extract Key Figures
- Use `visualizations/*.html` for interactive presentations
- Use `visualizations/*.png` for thesis document figures
- Reference `analysis/*.csv` for detailed statistics

### Step 4: Cite Results
All results include:
- Sample sizes for statistical power validation
- P-values for significance testing
- Effect sizes for practical significance
- Confidence intervals for uncertainty quantification

## 📚 For Academic Use

### Reproducibility
- All scripts are deterministic (except for random seed initialization)
- Complete parameter documentation
- Version-controlled codebase
- Comprehensive logging

### Statistical Rigor
- Multiple statistical tests for validation
- Effect size analysis for practical significance
- Confidence intervals for uncertainty quantification
- Proper multiple testing considerations

### Publication Quality
- Professional color schemes and formatting
- High-resolution figures (300 DPI)
- Interactive web-based visualizations
- Comprehensive data availability

## 🛠️ Dependencies

All scripts use standard scientific Python libraries:
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `matplotlib` - Static plotting
- `seaborn` - Statistical visualization
- `plotly` - Interactive visualization
- `scipy` - Statistical testing
- `scikit-learn` - Machine learning utilities

## 🤝 Support

For questions or issues with the research suite:
1. Check the console output for error messages
2. Verify input data file exists and is readable
3. Ensure all dependencies are installed
4. Review the generated log files for detailed information

## 📝 Notes

- All timestamps are in local time
- Results are deterministic for reproducibility
- Large datasets may require additional memory
- Interactive visualizations require a modern web browser

---

**🎓 Ready for thesis-quality research with comprehensive PSI controller evaluation!** 