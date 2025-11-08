import streamlit as st
import json
import pandas as pd
import numpy as np
import time
import random
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import plotly.graph_objects as go

# ==================== DATA STRUCTURES (Copied from gene.py for compatibility) ====================
# These dataclasses are necessary to reconstruct the Genotype objects from the JSON file.

@dataclass
class DevelopmentalGene:
    """Encodes developmental rules for phenotype construction"""
    rule_type: str
    trigger_condition: str
    parameters: Dict[str, float]

@dataclass
class ModuleGene:
    """Enhanced module gene with biological properties"""
    id: str
    module_type: str
    size: int
    activation: str
    normalization: str
    dropout_rate: float
    learning_rate_mult: float
    plasticity: float
    color: str
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)

@dataclass
class ConnectionGene:
    """Enhanced connection with synaptic properties"""
    source: str
    target: str
    weight: float
    connection_type: str
    delay: float
    plasticity_rule: str

@dataclass
class Genotype:
    """Complete genetic encoding with developmental program"""
    modules: List[ModuleGene]
    connections: List[ConnectionGene]
    developmental_rules: List[DevelopmentalGene] = field(default_factory=list)
    meta_parameters: Dict[str, float] = field(default_factory=dict)
    epigenetic_markers: Dict[str, float] = field(default_factory=dict)
    fitness: float = 0.0
    age: int = 0
    generation: int = 0
    lineage_id: str = ""
    parent_ids: List[str] = field(default_factory=list)
    accuracy: float = 0.0
    efficiency: float = 0.0
    complexity: float = 0.0
    robustness: float = 0.0
    form_id: int = 1

def dict_to_genotype(d: Dict) -> Genotype:
    """Deserializes a dictionary back into a Genotype object."""
    # Reconstruct nested dataclasses
    d['modules'] = [ModuleGene(**m) for m in d.get('modules', [])]
    d['connections'] = [ConnectionGene(**c) for c in d.get('connections', [])]
    d['developmental_rules'] = [DevelopmentalGene(**dr) for dr in d.get('developmental_rules', [])]
    
    # The Genotype dataclass can now be instantiated with the dictionary
    return Genotype(**d)

# ==================== REAL-WORLD TASK SIMULATION ====================

def simulate_task_performance(architecture: Genotype, task_name: str) -> Dict:
    """
    Simulates the performance of a given architecture on a specific 'real-world' task.
    This is a heuristic evaluation based on architectural properties.
    """
    # Architectural properties
    total_params = sum(m.size for m in architecture.modules)
    avg_plasticity = np.mean([m.plasticity for m in architecture.modules]) if architecture.modules else 0
    module_types = {m.module_type for m in architecture.modules}
    
    # Base performance score
    base_score = 0.5 + (architecture.accuracy - 0.5) * 0.5 # Anchor to evolved accuracy
    
    # Task-specific modifiers
    report = []
    if task_name == "Image Classification (CIFAR-100)":
        # Favors convolutional hierarchies
        conv_bonus = 0.2 if 'conv' in module_types else -0.1
        attention_bonus = 0.1 if 'attention' in module_types else 0.0
        efficiency_penalty = -0.1 * np.log1p(total_params / 1e6) # Penalize large models
        
        base_score += conv_bonus + attention_bonus + efficiency_penalty
        report.append(f"Convolutional Bonus: {conv_bonus:+.2f}")
        report.append(f"Attention Bonus: {attention_bonus:+.2f}")
        report.append(f"Size Penalty: {efficiency_penalty:+.2f}")

    elif task_name == "Abstract Reasoning (ARC Challenge)":
        # Favors graph, attention, and high plasticity
        graph_bonus = 0.25 if 'graph' in module_types else -0.1
        plasticity_bonus = 0.2 * (avg_plasticity - 0.5) # Reward high plasticity
        reasoning_bonus = 0.1 if 'logic_gate_array' in module_types or 'semantic_parser' in module_types else 0.0
        
        base_score += graph_bonus + plasticity_bonus + reasoning_bonus
        report.append(f"Graph/Relational Bonus: {graph_bonus:+.2f}")
        report.append(f"Plasticity Bonus: {plasticity_bonus:+.2f}")
        report.append(f"Symbolic Reasoning Bonus: {reasoning_bonus:+.2f}")

    elif task_name == "Language Modeling (LLM Benchmark)":
        # Favors attention, recurrence, and large parameter counts
        attention_bonus = 0.3 if 'attention' in module_types or 'transformer_block' in module_types else -0.2
        recurrent_bonus = 0.1 if 'recurrent' in module_types or 'lstm_unit' in module_types else 0.0
        scale_bonus = 0.15 * np.log1p(total_params / 1e7) # Reward very large models
        
        base_score += attention_bonus + recurrent_bonus + scale_bonus
        report.append(f"Attention/Transformer Bonus: {attention_bonus:+.2f}")
        report.append(f"Recurrent Bonus: {recurrent_bonus:+.2f}")
        report.append(f"Model Scale Bonus: {scale_bonus:+.2f}")

    elif task_name == "Robotics Control (Continuous Action)":
        # Favors recurrence, plasticity, and efficiency
        recurrent_bonus = 0.2 if 'recurrent' in module_types or 'liquid_network' in module_types else -0.1
        plasticity_bonus = 0.15 * avg_plasticity
        efficiency_bonus = 0.15 * architecture.efficiency # Reward low compute
        
        base_score += recurrent_bonus + plasticity_bonus + efficiency_bonus
        report.append(f"Recurrent/Temporal Bonus: {recurrent_bonus:+.2f}")
        report.append(f"Plasticity Bonus: {plasticity_bonus:+.2f}")
        report.append(f"Efficiency Bonus: {efficiency_bonus:+.2f}")

    final_score = np.clip(base_score + np.random.normal(0, 0.03), 0, 1) # Add noise and clip
    
    return {"score": final_score, "report": report}


# ==================== STREAMLIT APP ====================

def main():
    st.set_page_config(
        page_title="GENEVO Real-World Tester",
        layout="wide",
        page_icon="🤖"
    )

    # --- Custom CSS ---
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #0d3b66;
        text-align: center;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #64748b;
        text-align: center;
        margin-bottom: 2rem;
    }
    .st-emotion-cache-1vzeuhh { /* Target expander header */
        font-size: 1.1rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # --- Header ---
    st.markdown('<h1 class="main-header">GENEVO: Real-World Performance Simulator</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload your evolved architectures and test their mettle against simulated real-world benchmarks.</p>', unsafe_allow_html=True)

    # --- Session State Initialization ---
    if 'population' not in st.session_state:
        st.session_state.population = None

    # --- Sidebar for Upload and Selection ---
    st.sidebar.header("🔬 Experiment Setup")
    uploaded_file = st.sidebar.file_uploader(
        "Upload `genevo_experiment_data.json`",
        type="json",
        help="Upload the JSON file generated by the main GENEVO application."
    )

    if uploaded_file is not None:
        try:
            data = json.load(uploaded_file)
            pop_dicts = data.get('final_population', [])
            if pop_dicts:
                population = [dict_to_genotype(p) for p in pop_dicts]
                population.sort(key=lambda x: x.fitness, reverse=True)
                st.session_state.population = population
                st.sidebar.success(f"✅ Loaded {len(population)} architectures.")
            else:
                st.sidebar.error("JSON file does not contain a 'final_population' list.")
                st.session_state.population = None
        except Exception as e:
            st.sidebar.error(f"Error parsing JSON: {e}")
            st.session_state.population = None

    if st.session_state.population:
        st.sidebar.markdown("---")
        st.sidebar.header("🧠 Select Architecture")
        
        # Create options for the selectbox
        options = {
            f"Rank {i+1}: {ind.lineage_id} (Fitness: {ind.fitness:.4f})": ind.lineage_id
            for i, ind in enumerate(st.session_state.population)
        }
        
        selected_option = st.sidebar.selectbox(
            "Choose an architecture to test:",
            options.keys()
        )
        
        selected_lineage_id = options[selected_option]
        selected_arch = next((p for p in st.session_state.population if p.lineage_id == selected_lineage_id), None)

    else:
        st.info("Please upload a `genevo_experiment_data.json` file to begin.")
        st.stop()

    if not selected_arch:
        st.error("Could not find the selected architecture. Please try again.")
        st.stop()

    # --- Main Display Area ---
    st.header(f"🔬 Testing Dashboard: `{selected_arch.lineage_id}`")
    
    # Display key stats of the selected architecture
    with st.expander("Show Architecture Vitals", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Evolved Fitness", f"{selected_arch.fitness:.4f}")
        col2.metric("Evolved Accuracy", f"{selected_arch.accuracy:.3f}")
        col3.metric("Total Parameters", f"{sum(m.size for m in selected_arch.modules):,}")
        col4.metric("Complexity Score", f"{selected_arch.complexity:.3f}")

    st.markdown("---")

    # --- Task Simulation Section ---
    st.subheader("🚀 Real-World Task Simulation")

    with st.expander("🤔 How is this test performed? Is it just random?"):
        st.markdown("""
        Not at all! While there is a tiny bit of random noise to simulate real-world unpredictability, the score is primarily determined by a **heuristic, rule-based analysis** of the architecture's evolved properties. It works like this:

        1.  **Start with the Evolved Score:** The simulation begins with a baseline score derived from the architecture's final `accuracy` achieved during the evolution run. This grounds the test in its proven performance.

        2.  **Apply Task-Specific Bonuses & Penalties:** The system then inspects the architecture's "genes" (its modules and properties) and modifies the score based on how well they match the demands of the chosen task. For example:
            -   **Image Classification:** Gets a large bonus for having `conv` (Convolutional) modules, but a penalty for being excessively large (inefficient).
            -   **Abstract Reasoning:** Is rewarded for having `graph` modules and high `plasticity`, which are crucial for relational and flexible thinking.
            -   **Language Modeling:** Benefits greatly from `attention` or `transformer_block` modules and is rewarded for having a very large number of parameters (scale).
            -   **Robotics Control:** Favors `recurrent` or `liquid_network` modules for handling time-series data and gives a bonus for high `efficiency` (low computational cost).

        3.  **Add a Pinch of Noise:** A very small amount of random noise (`+/- 3%`) is added at the very end. This simulates the minor, unpredictable factors that always exist in real-world testing environments.

        **In short, this simulation isn't training the model.** It's a quick, analytical test to see if the principles of the evolved architecture align with the known principles of solving that type of problem. The detailed reports below show you exactly which factors contributed to the final score.
        """)

    st.markdown(
        "Click the button below to heuristically evaluate the selected architecture's suitability for various tasks "
        "based on its known properties (e.g., module types, plasticity, scale)."
    )

    tasks = [
        "Image Classification (CIFAR-100)",
        "Abstract Reasoning (ARC Challenge)",
        "Language Modeling (LLM Benchmark)",
        "Robotics Control (Continuous Action)"
    ]

    if 'benchmark_results' not in st.session_state:
        st.session_state.benchmark_results = {}

    if st.button("Run All Benchmark Simulations", type="primary"):
        st.session_state.benchmark_results = {} # Clear previous results
        progress_bar = st.progress(0, text="Starting benchmarks...")
        
        for i, task in enumerate(tasks):
            time.sleep(random.uniform(1.0, 2.5)) # Simulate work
            result = simulate_task_performance(selected_arch, task)
            st.session_state.benchmark_results[task] = result
            progress_bar.progress((i + 1) / len(tasks), text=f"Simulating {task}...")
        
        progress_bar.empty()
        st.success("All benchmark simulations complete!")

    if st.session_state.benchmark_results:
        st.markdown("---")
        st.subheader("📊 Benchmark Results")

        # --- AGI Scorecard ---
        st.markdown("#### 🏆 AGI Capabilities Scorecard")
        st.markdown("An aggregated view of the architecture's simulated performance across diverse domains.")

        scorecard_cols = st.columns(len(tasks))
        overall_scores = []
        domain_map = {
            "Image Classification (CIFAR-100)": "Perception",
            "Abstract Reasoning (ARC Challenge)": "Reasoning",
            "Language Modeling (LLM Benchmark)": "Language",
            "Robotics Control (Continuous Action)": "Action"
        }

        for i, task in enumerate(tasks):
            with scorecard_cols[i]:
                score = st.session_state.benchmark_results[task]['score']
                overall_scores.append(score)
                st.metric(label=f"**{domain_map[task]}**", value=f"{score:.1%}")

        # --- Overall Score Gauge ---
        avg_score = np.mean(overall_scores) if overall_scores else 0
        
        gauge_fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = avg_score * 100,
            title = {'text': "<b>Overall Performance Index</b>"},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "#0d3b66"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 50], 'color': '#fab3a9'},
                    {'range': [50, 80], 'color': '#f3e0b5'},
                    {'range': [80, 100], 'color': '#a9d1ab'}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        gauge_fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(gauge_fig, use_container_width=True)

        # --- Detailed Reports ---
        st.markdown("#### Detailed Task Reports")
        for task, result in st.session_state.benchmark_results.items():
            with st.expander(f"**{task}** - Performance Score: **{result['score']:.2f}**"):
                st.markdown("###### Performance Factors:")
                report_df = pd.DataFrame([line.split(': ') for line in result['report']], columns=['Factor', 'Impact'])
                st.table(report_df)
                st.markdown(
                    f"**Analysis:** The architecture's properties give it a simulated performance score of **{result['score']:.2f}** on this task. "
                    "This score is derived from its evolved accuracy, adjusted by bonuses and penalties specific to the task demands."
                )

    st.sidebar.markdown("---")
    st.sidebar.info(
        "This application provides a **heuristic simulation** of real-world performance. "
        "Scores are based on architectural properties, not on actual model training or inference."
    )

if __name__ == "__main__":
    main()
