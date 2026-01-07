# mr_utils.py
import numpy as np
import pandas as pd
import networkx as nx
from pyvis.network import Network

def build_knowledge_graph(df):    ## this function will be use for small pyvis html visualization

    """
    Builds a NetworkX graph from the original DataFrame.
    
    The graph will connect:
    (Company) -> [WORKS_IN] -> (Industry)
    (Company) -> [LOCATED_IN] -> (Region)
    """
    print("\n--- Building Knowledge Graph ---")
    
    G = nx.Graph()
    
    # We use a sample of the data (e.g., 100 unique companies)
    # to keep the graph visualization clean and readable.
    # A graph with 11,000 nodes would be too cluttered to see.
    companies_sample = df['CompanyName'].unique()[:100]
    df_sample = df[df['CompanyName'].isin(companies_sample)]
    
    for _, row in df_sample.iterrows():
        # Get node names
        company = row['CompanyName']
        industry = row['Industry']
        region = row['Region']
        
        # Add nodes with 'type' attribute for coloring
        G.add_node(company, type='company', title=f"Company: {company}")
        G.add_node(industry, type='industry', title=f"Industry: {industry}")
        G.add_node(region, type='region', title=f"Region: {region}")
        
        # Add edges (relationships)
        G.add_edge(company, industry, title='WORKS_IN')
        G.add_edge(company, region, title='LOCATED_IN')
        
    print(f"  - Knowledge Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G

def build_full_knowledge_graph(df):
    """
    Build a CLEAN Knowledge Graph for Node2Vec embeddings.
    No target leakage. No numeric feature nodes. 
    """
    print("\n--- Building Clean Knowledge Graph (for embeddings) ---")
    
    import networkx as nx
    G = nx.Graph()

    # Add only companies, industries, regions
    for _, row in df.iterrows():
        company = str(row['CompanyID'])
        industry = row['Industry']
        region = row['Region']

        # Add nodes
        G.add_node(company, type='company')
        G.add_node(industry, type='industry')
        G.add_node(region, type='region')

        # Add edges
        G.add_edge(company, industry)
        G.add_edge(company, region)

    print(f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G








# mr_utils.py
import pandas as pd
import networkx as nx
from pyvis.network import Network

# ... (your build_knowledge_graph function is above this) ...

# REPLACE your old visualize function with this one
def visualize_knowledge_graph(G, file_name="esg_knowledge_graph.html"):
    """
    Creates an interactive Pyvis visualization and returns the file path.
    """
    print(f"  - Generating interactive visualization... saving to {file_name}")
    
    # Create a Pyvis network
    net = Network(height="750px", width="100%", notebook=True, cdn_resources='in_line')
    
    # Load the NetworkX graph into Pyvis
    net.from_nx(G)
    
    # Add physics-based layout
    net.force_atlas_2based(gravity=-50, central_gravity=0.01, spring_length=100, spring_strength=0.08)
    
    # Show buttons for UI
    net.show_buttons(filter_=['physics'])
    
    try:
        net.show(file_name)
        print(f"  - Successfully saved to {file_name}.")
        # --- THIS IS THE NEW PART ---
        return file_name  # Return the path to the HTML file
    except Exception as e:
        print(f"Error during visualization: {e}")
        return None
    


    # mr_utils.py
import pandas as pd
import networkx as nx
from pyvis.network import Network

# ... (your build_knowledge_graph and visualize_knowledge_graph functions are above this) ...

def check_esg_rules(df):
    """
    Multi-region ESG rule base.
    Flags 'red-flag' companies whose financial strength (ProfitMargin)
    does not align with ESG performance, considering regional expectations.
    
    Region tiers:
        Europe: High expectation (score 3)
        North America, Oceania: Medium (score 2)
        Asia, Latin America, Middle East, Africa: Emerging (score 1)
    Returns: violators_df, compliance_violation_score (Series)
    """
    print("\n--- Checking Multi-Region ESG Rule ---")

    # Thresholds
    median_profit = df['ProfitMargin'].median()
    median_esg = df['ESG_Overall'].median()

    # Base conditions
    is_high_profit = df['ProfitMargin'] > median_profit
    is_low_esg = df['ESG_Overall'] < median_esg

    # Region tiers
    tier_1 = df['Region'].eq('Europe')
    tier_2 = df['Region'].isin(['North America', 'Oceania'])
    tier_3 = df['Region'].isin(['Asia', 'Latin America', 'Middle East', 'Africa'])

    # Combine
    conditions = [
        is_high_profit & is_low_esg & tier_1,
        is_high_profit & is_low_esg & tier_2,
        is_high_profit & is_low_esg & tier_3
    ]
    choices = [3, 2, 1]  # severity level
    compliance_score = np.select(conditions, choices, default=0)

    # Violator subset
    violators_df = df[is_high_profit & is_low_esg].copy()
    print(f"Using median ProfitMargin={median_profit:.2f}, ESG={median_esg:.2f}")
    print(f"Found {len(violators_df)} potential red-flag companies.")

    return violators_df, pd.Series(compliance_score, index=df.index, name='compliance_violation_score')





# mr_utils.py
# mr_utils.py
import pandas as pd
import networkx as nx
from pyvis.network import Network
import dowhy
from dowhy import CausalModel

# ... (your build_KG, visualize_KG, and check_rules functions are above this) ...

# --- REPLACE YOUR OLD FUNCTION WITH THIS ONE ---
# mr_utils.py
import pandas as pd
import networkx as nx
from pyvis.network import Network
import dowhy
from dowhy import CausalModel

# ... (your build_KG, visualize_KG, and check_rules functions are above this) ...

import pandas as pd
import networkx as nx
from pyvis.network import Network
import dowhy
from dowhy import CausalModel
import matplotlib.pyplot as plt
from node2vec import Node2Vec
import gensim

def estimate_causal_effect(df):
    """
    Defines a causal model and estimates the causal effect of ProfitMargin on ESG_Overall.
    Returns both the effect value and an updated dataframe including it.
    """
    print("\n--- Causal Inference Analysis ---")
    
    # Drop missing values for causal inference
    df_causal = df.dropna(subset=['ProfitMargin', 'ESG_Overall']).copy()

    # --- Define causal structure (graph hypothesis) ---
    graph_str = r"""
    digraph {
        Industry -> Revenue;
        Industry -> ProfitMargin;
        Region -> Revenue;
        Region -> ESG_Overall;
        Revenue -> ProfitMargin;
        Revenue -> MarketCap;
        Revenue -> ESG_Overall;
        MarketCap -> ESG_Overall;
        GrowthRate -> ESG_Overall;
        ProfitMargin -> ESG_Overall;
    }
    """
    print("  - Causal graph defined (Industry, Region, Revenue, etc.)")

    # --- Build and estimate the causal model ---
    from dowhy import CausalModel
    model = CausalModel(
        data=df_causal,
        treatment="ProfitMargin",
        outcome="ESG_Overall",
        graph=graph_str,
    )

    identified_effect = model.identify_effect(proceed_when_unidentifiable=True)
    estimate = model.estimate_effect(
        identified_effect,
        method_name="backdoor.linear_regression",
        test_significance=True,
    )

    print("\n--- Estimated Causal Effect ---")
    print(estimate)
    print(f"Estimated ATE: {estimate.value:.4f}")

    # --- Interpretation ---
    if estimate.value > 0:
        print(f"Interpretation: Increasing ProfitMargin causes ESG_Overall to increase by +{estimate.value:.4f}")
    else:
        print(f"Interpretation: Increasing ProfitMargin causes ESG_Overall to decrease by {estimate.value:.4f}")

    # --- Add this causal effect as a new column ---
    df['causal_ate_pm_to_esg'] = estimate.value
    print("✅ Added 'causal_ate_pm_to_esg' feature to dataframe")

    return df, estimate.value


def estimate_multiple_causal_effects(df, treatments=None):
    """
    Estimates causal effects for multiple variables on ESG_Overall.
    Adds new 'causal_ate_*' features to the dataframe.
    """
    from dowhy import CausalModel
    import numpy as np

    if treatments is None:
        treatments = ['ProfitMargin', 'Revenue', 'GrowthRate', 'MarketCap']

    df_causal = df.dropna(subset=['ESG_Overall']).copy()
    causal_results = {}

    for treat in treatments:
        if treat not in df_causal.columns:
            continue

        print(f"\n--- Estimating causal effect of {treat} on ESG_Overall ---")

        graph_str = f"""
        digraph {{
            Industry -> Revenue;
            Industry -> {treat};
            Region -> Revenue;
            Region -> ESG_Overall;
            Revenue -> {treat};
            Revenue -> MarketCap;
            Revenue -> ESG_Overall;
            MarketCap -> ESG_Overall;
            GrowthRate -> ESG_Overall;
            {treat} -> ESG_Overall;
        }}
        """

        try:
            model = CausalModel(
                data=df_causal,
                treatment=treat,
                outcome="ESG_Overall",
                graph=graph_str,
            )

            identified = model.identify_effect(proceed_when_unidentifiable=True)
            estimate = model.estimate_effect(
                identified,
                method_name="backdoor.linear_regression",
                test_significance=True,
            )

            df[f'causal_ate_{treat.lower()}_to_esg'] = estimate.value
            causal_results[treat] = estimate.value

            print(f"✅ {treat}: ATE={estimate.value:.4f}")

        except Exception as e:
            print(f"❌ Skipping {treat}: {e}")

    print("\n--- Summary of Causal Effects ---")
    for k, v in causal_results.items():
        print(f"{k} → ESG_Overall : {v:.4f}")

    return df, causal_results




# mr_utils.py
import pandas as pd
import networkx as nx
from pyvis.network import Network
import dowhy
from dowhy import CausalModel
import matplotlib.pyplot as plt  # <-- Add this import at the top

# ... (your other functions are above this) ...

# --- ADD THIS NEW FUNCTION (BASED ON YOUR CODE) ---
def draw_causal_diagram(file_name="causal_dag.png"):
    """
    Draws and saves the causal diagram (DAG) showing our
    hypothesized relationships.
    """
    print(f"\n--- Generating Causal Diagram (DAG) ---")
    
    G = nx.DiGraph()
    edges = [
        ("Industry", "Revenue"),
        ("Industry", "ProfitMargin"),
        ("Region", "Revenue"),
        ("Region", "ESG_Overall"),
        ("Revenue", "ProfitMargin"),
        ("Revenue", "MarketCap"),
        ("Revenue", "ESG_Overall"),
        ("MarketCap", "ESG_Overall"),
        ("GrowthRate", "ESG_Overall"),
        ("ProfitMargin", "ESG_Overall")
    ]
    G.add_edges_from(edges)

    # Define node positions for a cleaner layout
    pos = {
        'Industry': (0, 2), 'Region': (0, 0),
        'Revenue': (1, 2), 'ProfitMargin': (1, 1), 'MarketCap': (2, 2),
        'GrowthRate': (2, 0),
        'ESG_Overall': (3, 1)
    }
    
    plt.figure(figsize=(12, 8))
    nx.draw_networkx(
        G, 
        pos=pos,
        with_labels=True, 
        node_size=3000, 
        node_color="#A7C7E7", 
        font_size=10,
        arrowsize=20, 
        edge_color="gray"
    )
    plt.title("Causal DAG: Hypothesized Relationships for ESG Score")
    
    # --- We save the file instead of showing it ---
    try:
        plt.savefig(file_name)
        print(f"  - Successfully saved causal diagram to {file_name}")
    except Exception as e:
        print(f"Error saving plot: {e}")


        # mr_utils.py
import pandas as pd
import networkx as nx
from pyvis.network import Network
import dowhy
from dowhy import CausalModel
import matplotlib.pyplot as plt
from node2vec import Node2Vec  # <-- Add this import
import gensim # <-- Add this import
from node2vec import Node2Vec
import gensim
# ... (all your other functions are above this) ...

# ... (all your other functions are above this) ...

# --- Graph Embedding Utility (Node2Vec-based) ---
def generate_graph_embeddings(kg_graph, df, dimensions=8, walk_length=10, num_walks=20, workers=2):
    """
    Generates simple graph embeddings for each company node using Node2Vec.
    Args:
        kg_graph: NetworkX Graph (your knowledge graph)
        df: original dataframe with CompanyID column
        dimensions: embedding vector size
    Returns:
        A DataFrame with columns ['CompanyID', 'embed_1', ..., 'embed_k']
    """
    from node2vec import Node2Vec
    import pandas as pd
    import numpy as np

    print("\n--- Generating Graph Embeddings from Knowledge Graph ---")

    # Check that graph exists
    if kg_graph is None or len(kg_graph.nodes) == 0:
        raise ValueError("Knowledge Graph (kg_graph) is empty or not initialized.")

    # Initialize and train Node2Vec model
    node2vec = Node2Vec(kg_graph, dimensions=dimensions, walk_length=walk_length,
                        num_walks=num_walks, workers=workers)
    model = node2vec.fit(window=5, min_count=1, batch_words=4)

    # Create embedding dataframe for Company nodes only
    embed_list = []
    for company_id in df["CompanyID"].astype(str).unique():
        if company_id in model.wv:
            embed = model.wv[company_id]
        else:
            embed = np.zeros(dimensions)
        embed_list.append([company_id] + list(embed))

    embed_cols = ["CompanyID"] + [f"embed_{i+1}" for i in range(dimensions)]
    embed_df = pd.DataFrame(embed_list, columns=embed_cols)

    print(f"✅ Generated embeddings for {len(embed_df)} companies (dim={dimensions})")
    return embed_df



def visualize_combined_reasoning_graph(df, file_name="combined_reasoning_graph.html"):
    """
    Combines the Knowledge Graph (Company, Industry, Region)
    with causal edges (Industry→Revenue→ProfitMargin→ESG_Overall, etc.)
    into a single interactive Pyvis HTML graph.
    """
    import networkx as nx
    from pyvis.network import Network

    print("\n--- Building Combined Reasoning Graph (Knowledge + Causal) ---")

    G = nx.DiGraph()

    # === 1. Add Knowledge Graph edges ===
    for _, row in df.iterrows():
        company = str(row['CompanyID'])
        industry = row['Industry']
        region = row['Region']

        G.add_node(company, type='company', color='#A890FF', shape='dot', title=f"Company: {row['CompanyName']}")
        G.add_node(industry, type='industry', color='#FFA500', shape='triangle', title=f"Industry: {industry}")
        G.add_node(region, type='region', color='#00C0A3', shape='diamond', title=f"Region: {region}")

        # Add structural edges
        G.add_edge(company, industry, color='green', title='WORKS_IN')
        G.add_edge(company, region, color='green', title='LOCATED_IN')

    # === 2. Add conceptual causal edges ===
    causal_edges = [
        ("Industry", "Revenue"),
        ("Revenue", "ProfitMargin"),
        ("ProfitMargin", "ESG_Overall"),
        ("Region", "ESG_Overall"),
        ("GrowthRate", "ESG_Overall"),
        ("MarketCap", "ESG_Overall")
    ]

    causal_nodes = set([n for e in causal_edges for n in e])
    for node in causal_nodes:
        if node not in G.nodes:
            G.add_node(node, color='#FF6666', shape='ellipse', title=f"Causal Node: {node}")

    for src, dst in causal_edges:
        G.add_edge(src, dst, color='blue', title='CAUSAL_LINK')

    print(f"  - Added {len(causal_edges)} causal edges and {len(G.nodes)} total nodes.")

    # === 3. Create interactive visualization ===
    net = Network(height="750px", width="100%", directed=True, notebook=True, cdn_resources='in_line')
    net.from_nx(G)
    net.force_atlas_2based(gravity=-50, central_gravity=0.01, spring_length=100, spring_strength=0.08)
    net.show_buttons(filter_=['physics'])
    net.show(file_name)

    print(f"✅ Combined reasoning graph saved to {file_name}")
    return file_name



def visualize_combined_reasoning_graph3(df, compliance_score, file_name="combined_reasoning_graph3.html"):
    """
    Combines:
      1. Knowledge Graph (Company, Industry, Region)
      2. Causal Graph (Industry→Revenue→ProfitMargin→ESG_Overall)
      3. Rule Layer (Red-flag companies based on compliance_score)
    into a single interactive Pyvis HTML graph.
    """
    import networkx as nx
    from pyvis.network import Network

    print("\n--- Building Combined Reasoning Graph (Knowledge + Causal + Rules) ---")

    G = nx.DiGraph()

    # === 1. Add Knowledge Graph edges (Company–Industry–Region) ===
    for _, row in df.iterrows():
        company = str(row['CompanyID'])
        industry = row['Industry']
        region = row['Region']

        # Define red-flag color based on compliance score
        if compliance_score.loc[_] > 0:
            node_color = '#FF4444'  # 🔴 Red for violators
        else:
            node_color = '#A890FF'  # 🟣 Default company

        G.add_node(company, type='company', color=node_color, shape='dot', title=f"Company: {row['CompanyName']}")
        G.add_node(industry, type='industry', color='#FFA500', shape='triangle', title=f"Industry: {industry}")
        G.add_node(region, type='region', color='#00C0A3', shape='diamond', title=f"Region: {region}")

        # Add structural (KG) edges
        G.add_edge(company, industry, color='green', title='WORKS_IN')
        G.add_edge(company, region, color='green', title='LOCATED_IN')

    # === 2. Add conceptual causal edges ===
    causal_edges = [
        ("Industry", "Revenue"),
        ("Revenue", "ProfitMargin"),
        ("ProfitMargin", "ESG_Overall"),
        ("Region", "ESG_Overall"),
        ("GrowthRate", "ESG_Overall"),
        ("MarketCap", "ESG_Overall")
    ]
    causal_nodes = set([n for e in causal_edges for n in e])
    for node in causal_nodes:
        if node not in G.nodes:
            G.add_node(node, color='#FF6666', shape='ellipse', title=f"Causal Node: {node}")

    for src, dst in causal_edges:
        G.add_edge(src, dst, color='blue', title='CAUSAL_LINK')

    print(f"  - Added {len(causal_edges)} causal edges and {len(G.nodes)} total nodes.")

    # === 3. Create interactive visualization ===
    net = Network(height="750px", width="100%", directed=True, notebook=True, cdn_resources='in_line')
    net.from_nx(G)
    net.force_atlas_2based(gravity=-50, central_gravity=0.01, spring_length=100, spring_strength=0.08)
    net.show_buttons(filter_=['physics'])
    net.show(file_name)

    print(f"✅ Combined reasoning graph with rule-based violators saved to {file_name}")
    return file_name
def visualize_combined_reasoning_graph4(
    df,
    compliance_score=None,      # <- your median or dynamic rule column (Series)
    file_name="combined_reasoning_graph3.html"
):
    """
    Visualizes:
      1. Knowledge Graph (Company–Industry–Region)
      2. Causal Graph (Industry→Revenue→ProfitMargin→ESG_Overall, etc.)
      3. Rule Layer (violator severity coloring)

    Produces an interactive PyVis HTML graph.
    """

    import networkx as nx
    from pyvis.network import Network

    print("\n--- Building Combined Reasoning Graph (Knowledge + Causal + Rules) ---")

    # ----------------------------------------------------------
    # 0. PREPARE RULE COLUMN
    # ----------------------------------------------------------
    if compliance_score is None:
        print("⚠ No compliance score provided → all companies shown as normal.")
        df["rule_value"] = 0
    else:
        df["rule_value"] = compliance_score

    # Severity color map
    severity_colors = {
        0: "#4A89FF",  # normal (blue)
        1: "#FFF200",  # mild (yellow)
        2: "#FF9900",  # medium (orange)
        3: "#FF0000",  # red (severe)
    }

    # ----------------------------------------------------------
    # 1. BUILD GRAPH
    # ----------------------------------------------------------
    G = nx.DiGraph()

    # Add company, industry, region nodes
    for _, row in df.iterrows():
        company = str(row["CompanyID"])
        industry = row["Industry"]
        region = row["Region"]
        sev = int(row["rule_value"])

        # color based on rule severity
        node_color = severity_colors.get(sev, "#4A89FF")

        # COMPANY NODE
        G.add_node(
            company,
            type="company",
            color=node_color,
            shape="dot",
            title=(
                f"<b>Company:</b> {row['CompanyName']}<br>"
                f"<b>Severity:</b> {sev}<br>"
                f"<b>ProfitMargin:</b> {row['ProfitMargin']}<br>"
                f"<b>ESG_Overall:</b> {row['ESG_Overall']}<br>"
                f"<b>Region:</b> {region}<br>"
            )
        )

        # INDUSTRY NODE
        G.add_node(
            industry,
            type="industry",
            color="#FFA853",
            shape="triangle",
            title=f"Industry: {industry}"
        )

        # REGION NODE
        G.add_node(
            region,
            type="region",
            color="#00B3A3",
            shape="diamond",
            title=f"Region: {region}"
        )

        # EDGES
        G.add_edge(company, industry, color="green", title="WORKS_IN")
        G.add_edge(company, region, color="green", title="LOCATED_IN")

    # ----------------------------------------------------------
    # 2. ADD CAUSAL EDGES
    # ----------------------------------------------------------
    causal_edges = [
        ("Industry", "Revenue"),
        ("Revenue", "ProfitMargin"),
        ("ProfitMargin", "ESG_Overall"),
        ("Region", "ESG_Overall"),
        ("GrowthRate", "ESG_Overall"),
    ]

    causal_nodes = set([n for e in causal_edges for n in e])
    for node in causal_nodes:
        if node not in G.nodes:
            G.add_node(
                node,
                type="causal",
                color="#FF6666",
                shape="ellipse",
                title=f"Causal Node: {node}"
            )

    for src, dst in causal_edges:
        G.add_edge(src, dst, color="blue", title="CAUSAL_LINK")

    print(f"✓ Added {len(causal_edges)} causal edges and {len(G.nodes)} total nodes.")

    # ----------------------------------------------------------
    # 3. RENDER INTERACTIVE GRAPH
    # ----------------------------------------------------------
    net = Network(
        height="750px",
        width="100%",
        directed=True,
        notebook=True,
        cdn_resources="in_line"
    )
    net.from_nx(G)

    # Physics for layout
    net.force_atlas_2based(
        gravity=-50,
        central_gravity=0.01,
        spring_length=100,
        spring_strength=0.08
    )

    net.show(file_name)
    print(f"\n📁 Combined reasoning graph saved to {file_name}")
    return file_name
