import streamlit as st
import json
import pandas as pd
import numpy as np
import time
import random
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
from collections import Counter


# ==================== DATA STRUCTURES & CORE FUNCTIONS (Copied from gene.py for compatibility) ====================
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
    # We need to handle cases where new fields were added to Genotype but are not in the JSON
    # by checking for their existence before passing to the constructor.
    known_fields = {f.name for f in Genotype.__dataclass_fields__.values()}
    filtered_d = {k: v for k, v in d.items() if k in known_fields}

    return Genotype(**filtered_d)

def is_viable(genotype: Genotype) -> bool:
    """
    Checks if a genotype is structurally viable.
    """
    if not genotype.modules or not genotype.connections:
        return False

    G = nx.DiGraph()
    module_ids = {m.id for m in genotype.modules}
    
    for conn in genotype.connections:
        if conn.source in module_ids and conn.target in module_ids:
            G.add_edge(conn.source, conn.target)

    if G.number_of_nodes() < 2: return False

    input_nodes = [node for node, in_degree in G.in_degree() if in_degree == 0]
    output_nodes = [node for node, out_degree in G.out_degree() if out_degree == 0]

    if not input_nodes:
        potential_inputs = [m.id for m in genotype.modules if 'input' in m.id or 'embed' in m.id or 'V1' in m.id]
        input_nodes = [node for node in potential_inputs if node in G.nodes]

    if not output_nodes:
        potential_outputs = [m.id for m in genotype.modules if 'output' in m.id or 'PFC' in m.id]
        output_nodes = [node for node in potential_outputs if node in G.nodes]

    if not input_nodes or not output_nodes: return False

    for start_node in input_nodes:
        for end_node in output_nodes:
            if start_node in G and end_node in G and nx.has_path(G, start_node, end_node):
                return True

    return Genotype(**d)

# ==================== REAL-WORLD TASK SIMULATION ====================

def simulate_task_performance(architecture: Genotype, task_name: str) -> Dict:
    """
    Simulates the performance of a given architecture on a specific 'real-world' task.
    This is a heuristic evaluation based on architectural properties.
    """
    # Use the full, rigorous evaluation function from gene.py
    # This provides a much more nuanced score.
    fitness, scores = evaluate_fitness(architecture, task_name, architecture.generation)
    
    # Create a report based on the component scores
    report = [
        f"Task Accuracy Score: {scores['task_accuracy']:.3f}",
        f"Efficiency Score: {scores['efficiency']:.3f}",
        f"Robustness Score: {scores['robustness']:.3f}",
        f"Generalization Score: {scores['generalization']:.3f}"
    ]
    
    return {"score": fitness, "report": report, "components": scores}

# ==================== CORE ANALYSIS FUNCTIONS (from gene.py) ====================
# These functions are copied directly from gene.py to provide the same deep analysis capabilities.

def evaluate_fitness(genotype: Genotype, task_type: str, generation: int, weights: Optional[Dict[str, float]] = None, **kwargs) -> Tuple[float, Dict[str, float]]:
    """
    Multi-objective fitness evaluation with realistic task simulation.
    This is the same function as in gene.py for consistency.
    """
    scores = {'task_accuracy': 0.0, 'efficiency': 0.0, 'robustness': 0.0, 'generalization': 0.0}
    total_params = sum(m.size for m in genotype.modules)
    avg_plasticity = np.mean([m.plasticity for m in genotype.modules]) if genotype.modules else 0
    connection_density = len(genotype.connections) / (len(genotype.modules) ** 2 + 1) if genotype.modules else 0

    # Task-specific accuracy simulation
    if 'ARC' in task_type:
        graph_attention_count = sum(1 for m in genotype.modules if m.module_type in ['graph', 'attention'])
        compositional_score = graph_attention_count / (len(genotype.modules) + 1e-6)
        plasticity_bonus = avg_plasticity * 0.4
        efficiency_penalty = np.exp(-total_params / 50000)
        scores['task_accuracy'] = (compositional_score * 0.4 + plasticity_bonus * 0.3 + efficiency_penalty * 0.3 + np.random.normal(0, 0.05))
    elif 'Image' in task_type:
        conv_count = sum(1 for m in genotype.modules if m.module_type == 'conv')
        hierarchical_bonus = 0.2 if genotype.form_id in [1, 4] else 0.0
        scores['task_accuracy'] = ((conv_count / (len(genotype.modules) + 1e-6)) * 0.5 + hierarchical_bonus + connection_density * 0.2 + np.random.normal(0, 0.05))
    elif 'Language' in task_type:
        attn_count = sum(1 for m in genotype.modules if 'attention' in m.module_type or 'transformer' in m.module_type)
        depth_bonus = len(genotype.modules) / 10
        scores['task_accuracy'] = ((attn_count / (len(genotype.modules) + 1e-6)) * 0.6 + min(depth_bonus, 0.3) + np.random.normal(0, 0.05))
    elif 'Robotics' in task_type or 'Sequential' in task_type:
        rec_count = sum(1 for m in genotype.modules if 'recurrent' in m.module_type or 'liquid' in m.module_type)
        memory_bonus = 0.3 if any('memory' in m.id for m in genotype.modules) else 0.0
        scores['task_accuracy'] = ((rec_count / (len(genotype.modules) + 1e-6)) * 0.5 + memory_bonus + avg_plasticity * 0.15 + np.random.normal(0, 0.05))

    scores['task_accuracy'] = np.clip(scores['task_accuracy'], 0, 1)
    
    # Efficiency score
    param_efficiency = 1.0 / (1.0 + np.log(1 + total_params / 10000))
    connection_efficiency = 1.0 - min(connection_density, 0.8)
    scores['efficiency'] = (param_efficiency + connection_efficiency) / 2
    
    # Robustness score
    robustness_from_diversity = len(set(c.connection_type for c in genotype.connections)) / 3 if genotype.connections else 0
    robustness_from_plasticity = 1.0 - abs(avg_plasticity - 0.5) * 2
    scores['robustness'] = (robustness_from_diversity * 0.5 + robustness_from_plasticity * 0.5)
    
    # Generalization potential
    depth = len(genotype.modules)
    modularity_score = 1.0 - abs(connection_density - 0.3) * 2
    scores['generalization'] = (min(depth / 10, 1.0) * 0.4 + modularity_score * 0.3 + avg_plasticity * 0.3)

    if weights is None:
        weights = {'task_accuracy': 0.6, 'efficiency': 0.2, 'robustness': 0.1, 'generalization': 0.1}
    
    total_fitness = sum(scores[k] * weights[k] for k in weights)
    return max(total_fitness, 1e-6), scores

def analyze_lesion_sensitivity(architecture: Genotype, base_fitness: float, task_type: str, fitness_weights: Dict) -> Dict[str, float]:
    criticality_scores = {}
    for module in architecture.modules:
        if 'input' in module.id or 'output' in module.id: continue
        lesioned_arch = architecture.copy()
        lesioned_arch.modules = [m for m in lesioned_arch.modules if m.id != module.id]
        lesioned_arch.connections = [c for c in lesioned_arch.connections if c.source != module.id and c.target != module.id]
        if not is_viable(lesioned_arch): continue
        lesioned_fitness, _ = evaluate_fitness(lesioned_arch, task_type, lesioned_arch.generation, fitness_weights)
        criticality_scores[f"Module: {module.id}"] = base_fitness - lesioned_fitness
    return criticality_scores

def analyze_information_flow(architecture: Genotype) -> Dict[str, float]:
    G = nx.DiGraph()
    for module in architecture.modules: G.add_node(module.id)
    for conn in architecture.connections:
        if conn.weight > 1e-6: G.add_edge(conn.source, conn.target, weight=1.0/conn.weight)
    if not G.nodes: return {}
    return nx.betweenness_centrality(G, weight='weight', normalized=True)

def generate_pytorch_code(architecture: Genotype) -> str:
    module_defs = [f"            # Fitness: {architecture.fitness:.4f}, Accuracy: {architecture.accuracy:.4f}"]
    for m in architecture.modules:
        if m.module_type == 'mlp': module_defs.append(f"            '{m.id}': nn.Sequential(nn.Linear({m.size}, {m.size}), nn.GELU()),")
        elif m.module_type == 'attention': module_defs.append(f"            '{m.id}': nn.MultiheadAttention(embed_dim={m.size}, num_heads=8, batch_first=True),")
        elif m.module_type == 'conv': module_defs.append(f"            '{m.id}': nn.Conv2d(3, {m.size}, 3, padding=1),")
        elif m.module_type == 'recurrent': module_defs.append(f"            '{m.id}': nn.LSTM({m.size}, {m.size}, batch_first=True),")
        else: module_defs.append(f"            '{m.id}': nn.Identity(), # Placeholder for '{m.module_type}'")
    
    G = nx.DiGraph([(c.source, c.target) for c in architecture.connections])
    try: exec_order = list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible: exec_order = [m.id for m in architecture.modules]

    forward_pass = ["        outputs = {} # Dict to store module outputs"]
    # Find true inputs (no incoming connections)
    true_inputs = [m.id for m in architecture.modules if G.in_degree(m.id) == 0]
    if not true_inputs: true_inputs = [exec_order[0]] # Fallback
    for in_node in true_inputs: forward_pass.append(f"        outputs['{in_node}'] = x # Feed input to '{in_node}'")

    for mid in exec_order:
        if mid in true_inputs: continue
        inputs = [c.source for c in architecture.connections if c.target == mid]
        if not inputs: continue
        
        input_str = " + ".join([f"outputs['{i}']" for i in inputs if i in exec_order])
        if not input_str: input_str = 'x' # Fallback

        module_type = next((m.module_type for m in architecture.modules if m.id == mid), '')
        if module_type == 'recurrent': forward_pass.append(f"        out, _ = self.evolved_modules['{mid}']({input_str}); outputs['{mid}'] = out")
        elif module_type == 'attention': forward_pass.append(f"        attn_out, _ = self.evolved_modules['{mid}']({input_str}, {input_str}, {input_str}); outputs['{mid}'] = attn_out")
        else: forward_pass.append(f"        outputs['{mid}'] = self.evolved_modules['{mid}']({input_str})")

    # Find true output (no outgoing connections)
    true_output = [m.id for m in architecture.modules if G.out_degree(m.id) == 0]
    if not true_output: true_output = [exec_order[-1]] # Fallback
    forward_pass.append(f"        return outputs['{true_output[0]}']")

    return f"""
import torch
import torch.nn as nn

class EvolvedArchitecture(nn.Module):
    def __init__(self):
        super().__init__()
        self.evolved_modules = nn.ModuleDict({{
{chr(10).join(module_defs)}
        }})

    def forward(self, x):
{chr(10).join(forward_pass)}
""".strip()

def generate_tensorflow_code(architecture: Genotype) -> str:
    # This is a simplified version for brevity. A full implementation would be similar to the PyTorch one.
    return f"# TensorFlow code generation for {architecture.lineage_id} is a work in progress.\n# Key properties: {len(architecture.modules)} modules, {len(architecture.connections)} connections."

def visualize_genotype_2d(genotype: Genotype) -> go.Figure:
    G = nx.DiGraph()
    for module in genotype.modules:
        G.add_node(module.id, color=module.color, size=15 + np.sqrt(module.size),
                   hover_text=f"<b>{module.id}</b><br>Type: {module.module_type}<br>Size: {module.size}")
    for conn in genotype.connections:
        if conn.source in G.nodes and conn.target in G.nodes: G.add_edge(conn.source, conn.target)
            
    try: pos = nx.kamada_kawai_layout(G)
    except Exception: pos = nx.spring_layout(G, seed=42)

    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]; x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None]); edge_y.extend([y0, y1, None])
    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color='#888'), hoverinfo='none', mode='lines')

    node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x); node_y.append(y)
        node_text.append(G.nodes[node]['hover_text'])
        node_color.append(G.nodes[node]['color'])
        node_size.append(G.nodes[node]['size'])

    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text', text=[node for node in G.nodes()],
                            textposition="top center", hoverinfo='text', hovertext=node_text,
                            marker=dict(showscale=False, color=node_color, size=node_size, line=dict(width=2, color='black')))
    
    fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(title=f"<b>2D View: {genotype.lineage_id}</b>", title_x=0.5, showlegend=False,
                             margin=dict(b=20, l=5, r=5, t=50), xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                             yaxis=dict(showgrid=False, zeroline=False, showticklabels=False), height=400, plot_bgcolor='white'))
    return fig

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
        color: #1e3a8a; /* Darker blue */
        text-align: center;
    }
    .sub-header {
        font-size: 1.1rem;
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
    st.markdown('<h1 class="main-header">GENEVO: Architecture Analysis & Benchmarking</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">A rigorous diagnostic tool to perform deep analysis and simulated benchmarking on evolved architectures.</p>', unsafe_allow_html=True)

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
    st.header(f"🔬 Analysis Dashboard: `{selected_arch.lineage_id}`")
    
    # Display key stats of the selected architecture
    tab_vitals, tab_analysis, tab_benchmark, tab_code = st.tabs([
        "🌐 Vitals & Architecture", 
        "🔬 Causal & Structural Analysis",
        "🚀 Simulated Benchmarking",
        "💻 Code Export"
    ])

    # --- TAB 1: Vitals & Architecture ---
    with tab_vitals:
        vitals_col1, vitals_col2 = st.columns([1, 2])
        with vitals_col1:
            st.markdown("#### Quantitative Profile")
            st.metric("Evolved Fitness", f"{selected_arch.fitness:.4f}")
            st.metric("Evolved Accuracy", f"{selected_arch.accuracy:.3f}")
            st.metric("Efficiency Score", f"{selected_arch.efficiency:.3f}")
            st.metric("Robustness Score", f"{selected_arch.robustness:.3f}")
            st.metric("Total Parameters", f"{sum(m.size for m in selected_arch.modules):,}")
            st.metric("Complexity Score", f"{selected_arch.complexity:.3f}")

        with vitals_col2:
            st.markdown("#### Architectural Blueprint (2D)")
            st.plotly_chart(visualize_genotype_2d(selected_arch), use_container_width=True)

    # --- TAB 2: Causal & Structural Analysis ---
    with tab_analysis:
        st.markdown("This tab dissects the functional importance of the architecture's components using techniques from `gene.py`.")
        
        if st.button("Run Full Causal Analysis", key="run_causal_analysis"):
            st.session_state.causal_results = {}
            
            with st.spinner("Performing lesion sensitivity analysis..."):
                # Use default weights for analysis
                fitness_weights = {'task_accuracy': 0.6, 'efficiency': 0.2, 'robustness': 0.1, 'generalization': 0.1}
                # The task type here is less critical, it's for the fitness function context
                task_type_for_eval = "Abstract Reasoning (ARC-AGI-2)"
                
                criticality_scores = analyze_lesion_sensitivity(
                    selected_arch, selected_arch.fitness, task_type_for_eval, fitness_weights
                )
                st.session_state.causal_results['criticality'] = sorted(criticality_scores.items(), key=lambda item: item[1], reverse=True)

            with st.spinner("Analyzing information flow..."):
                centrality_scores = analyze_information_flow(selected_arch)
                st.session_state.causal_results['centrality'] = sorted(centrality_scores.items(), key=lambda item: item[1], reverse=True)

        if 'causal_results' in st.session_state and st.session_state.causal_results:
            causal_col1, causal_col2 = st.columns(2)
            with causal_col1:
                st.subheader("Lesion Sensitivity")
                st.markdown("Components whose removal causes the largest drop in fitness.")
                crit_data = st.session_state.causal_results.get('criticality', [])
                if crit_data:
                    df_crit = pd.DataFrame(crit_data, columns=['Component', 'Fitness Drop'])
                    st.dataframe(df_crit.head(10))
                else:
                    st.info("No criticality data available.")

            with causal_col2:
                st.subheader("Information Flow Backbone")
                st.markdown("Modules with the highest betweenness centrality, crucial for routing information.")
                cent_data = st.session_state.causal_results.get('centrality', [])
                if cent_data:
                    df_cent = pd.DataFrame(cent_data, columns=['Module', 'Centrality Score'])
                    st.dataframe(df_cent.head(10))
                else:
                    st.info("No centrality data available.")

    # --- TAB 3: Simulated Benchmarking ---
    with tab_benchmark:
        st.subheader("🚀 Real-World Task Simulation")
        with st.expander("🤔 How is this test performed?"):
            st.markdown("""
            This simulation uses the **exact same `evaluate_fitness` function from `gene.py`**. It is a rigorous, rule-based analysis of the architecture's properties against different task demands.

            1.  **Task-Specific Heuristics:** Each task (e.g., Vision, Language) has rules that reward specific architectural features (e.g., `conv` modules for vision, `attention` for language).
            2.  **Multi-Objective Score:** The final score is a weighted sum of four components:
                - **Task Accuracy:** The heuristic score for the specific task.
                - **Efficiency:** A penalty for high parameter counts and connection density.
                - **Robustness:** A measure of architectural stability.
                - **Generalization:** A score based on properties linked to generalization potential.
            
            This provides a much more nuanced and credible estimate of performance than a simple random score. The detailed reports show the breakdown of these components.
            """)

        tasks = ["Vision (ImageNet)", "Language (MMLU-Pro)", "Abstract Reasoning (ARC-AGI-2)", "Robotics Control (Continuous Action)"]

        if 'benchmark_results' not in st.session_state:
            st.session_state.benchmark_results = {}

        if st.button("Run All Benchmark Simulations", type="primary"):
            st.session_state.benchmark_results = {}
            progress_bar = st.progress(0, text="Starting benchmarks...")
            for i, task in enumerate(tasks):
                time.sleep(1.5) # Simulate work
                result = simulate_task_performance(selected_arch, task)
                st.session_state.benchmark_results[task] = result
                progress_bar.progress((i + 1) / len(tasks), text=f"Simulating {task}...")
            progress_bar.empty()
            st.success("All benchmark simulations complete!")

        if st.session_state.benchmark_results:
            st.markdown("---")
            st.subheader("📊 Benchmark Results")
            
            for task, result in st.session_state.benchmark_results.items():
                with st.expander(f"**{task}** - Overall Score: **{result['score']:.3f}**"):
                    st.markdown("###### Component Scores:")
                    cols = st.columns(4)
                    cols[0].metric("Task Accuracy", f"{result['components']['task_accuracy']:.3f}")
                    cols[1].metric("Efficiency", f"{result['components']['efficiency']:.3f}")
                    cols[2].metric("Robustness", f"{result['components']['robustness']:.3f}")
                    cols[3].metric("Generalization", f"{result['components']['generalization']:.3f}")

    # --- TAB 4: Code Export ---
    with tab_code:
        st.markdown("The genotype can be translated into functional code for deep learning frameworks, providing a direct path from discovery to application.")
        code_col1, code_col2 = st.columns(2)
        with code_col1:
            st.subheader("PyTorch Code")
            st.code(generate_pytorch_code(selected_arch), language='python')
        with code_col2:
            st.subheader("TensorFlow / Keras Code")
            st.code(generate_tensorflow_code(selected_arch), language='python')

    st.sidebar.markdown("---")
    st.sidebar.info(
        "This application provides a **rigorous analysis and heuristic simulation** of real-world performance. "
        "Scores are based on architectural properties from `gene.py`'s evaluation logic, not on actual model training."
    )

if __name__ == "__main__":
    main()
