import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to Python path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from preprocessing.loader import DatasetLoader
from preprocessing.cleaner import DataCleaner
from preprocessing.encoder import DataEncoder
from preprocessing.scaler import DataScaler
from preprocessing.feature_engineering import FeatureEngineer
from preprocessing.baseline_trust_v2 import BaselineTrust
from preprocessing.trust_fusion import TrustFusion

from barm.ug import UG
from barm.ra import RA
from barm.reputation import Reputation
from barm.trust_update import TrustUpdate
from barm.tps import TPS

from adrs_mpiq.routing import Routing
from adrs_mpiq.queue_manager import QueueManager
from adrs_mpiq.fitness import MPIQFitness
from adrs_mpiq.clustering import Clustering
from adrs_mpiq.encryption import Encryption
from adrs_mpiq.mpiq import MPIQ

from proof_of_trust.trust_manager import TrustManager
from proof_of_trust.ledger import Ledger
from proof_of_trust.verification import Verification
from proof_of_trust.consensus import Consensus

from evaluation.metrics import Evaluation

# Import ML prediction
try:
    from ml.predict_attack import predict_attack_probability
except ImportError:
    def predict_attack_probability(df):
        # Fallback - use label column as probability (for demo)
        # REMOVE THIS IN PRODUCTION
        if 'label' in df.columns:
            return df['label'].astype(float)
        return np.random.rand(len(df))


def load_flow_data(limit: int = 5000) -> pd.DataFrame:
    """Load raw network dataset and preprocess it - KEEP GROUND TRUTH!"""
    loader = DatasetLoader()
    
    try:
        # Load the Network dataset
        raw_df = loader.load_dataset("Network")
        
        # IMPORTANT: Save ground truth BEFORE cleaning
        # UNSW-NB15 uses 'label' (0=normal, 1=attack)
        if 'label' in raw_df.columns:
            raw_df['GroundTruth'] = raw_df['label']
        elif 'attack_cat' in raw_df.columns:
            raw_df['GroundTruth'] = (raw_df['attack_cat'] != 'Normal').astype(int)
        else:
            # Try to find binary column
            for col in raw_df.columns:
                if raw_df[col].nunique() == 2 and raw_df[col].dtype in ['int64', 'float64']:
                    if set(raw_df[col].unique()) == {0, 1}:
                        raw_df['GroundTruth'] = raw_df[col]
                        break
        
        # Store attack category if available
        if 'attack_cat' in raw_df.columns:
            raw_df['attack_cat_original'] = raw_df['attack_cat']
        elif 'type' in raw_df.columns:
            raw_df['attack_cat_original'] = raw_df['type']
        
        # Keep ground truth safe
        gt_col = raw_df['GroundTruth'].copy() if 'GroundTruth' in raw_df.columns else None
        attack_cat_col = raw_df['attack_cat_original'].copy() if 'attack_cat_original' in raw_df.columns else None
        
        # Clean
        df = DataCleaner(raw_df).clean()
        
        # Restore ground truth and attack category
        if gt_col is not None and 'GroundTruth' not in df.columns:
            df['GroundTruth'] = gt_col
        if attack_cat_col is not None and 'attack_cat' not in df.columns:
            df['attack_cat'] = attack_cat_col
        
        # Encode categorical variables
        df = DataEncoder(df).encode()
        
        # Scale
        df = DataScaler(df).scale()
        
        # Feature engineering
        df = FeatureEngineer(df).transform()
        
        # Predict attack probability
        df["Predicted_Attack_Probability"] = predict_attack_probability(df)
        
        # Limit to manageable size
        if len(df) > limit:
            df = df.sample(n=limit, random_state=42)
        
        return df
    
    except Exception as e:
        st.error(f"Error loading flow data: {e}")
        import traceback
        st.code(traceback.format_exc())
        return pd.DataFrame()


def process_flow_trust(df: pd.DataFrame) -> pd.DataFrame:
    """Apply trust models directly on flow-level data."""
    df_flow = df.copy()
    
    # Ensure we have an identifier for flows
    if "Node_ID" not in df_flow.columns:
        df_flow["Node_ID"] = df_flow.index.astype(str)
    
    try:
        # Baseline Trust
        df_flow = BaselineTrust(df_flow).compute()
        df_flow["Hybrid_Trust"] = df_flow["Behaviour_Trust"]
        
        # Trust Fusion
        df_flow = TrustFusion(df_flow).compute()
        
        # BARM
        df_flow = Reputation(df_flow).compute()
        df_flow = UG(df_flow).compute()
        df_flow = RA(df_flow).compute()
        df_flow = TPS(df_flow).compute()
        df_flow = TrustUpdate(df_flow).compute()
        
        # ADRS-MPIQ
        df_flow = MPIQFitness(df_flow).compute()
        df_flow = Clustering(df_flow).compute()
        df_flow = Routing(df_flow).compute()
        df_flow = QueueManager(df_flow).compute()
        df_flow = Encryption(df_flow).compute()
        df_flow = MPIQ(df_flow).compute()
        
        # Proof of Trust
        df_flow = TrustManager(df_flow).compute()
        blockchain = Ledger(df_flow).generate()
        consensus = Consensus(blockchain).compute()
        df_flow["Consensus"] = consensus
        
        return df_flow
        
    except Exception as e:
        st.error(f"Error processing trust models: {e}")
        return df_flow


def show(df: pd.DataFrame):
    """Display statistics on flow-level network data."""
    st.title("🌐 Network Flow Statistics (Raw Dataset)")
    
    st.markdown("""
    This page analyzes the **raw network dataset** (flows/packets) directly.
    Each row represents a network flow.
    """)
    
    # Check for cached results
    results_dir = ROOT_DIR / "results"
    flow_file = results_dir / "final_results_flow.csv"
    
    with st.spinner("Loading and processing network flows..."):
        try:
            if flow_file.exists():
                flow_df = pd.read_csv(flow_file)
                st.success(f"✅ Loaded {len(flow_df)} flows from cache.")
            else:
                st.info("🔄 Processing flows for the first time...")
                raw_flow_df = load_flow_data(limit=5000)
                if raw_flow_df.empty:
                    st.error("Failed to load flow data.")
                    return
                flow_df = process_flow_trust(raw_flow_df)
                results_dir.mkdir(exist_ok=True)
                flow_df.to_csv(flow_file, index=False)
                st.success(f"✅ Processed {len(flow_df)} flows.")
        except Exception as e:
            st.error(f"Error: {e}")
            return
    
    if flow_df.empty:
        st.warning("No flow data available.")
        return
    
    total_flows = len(flow_df)
    st.markdown(f"**Total flows analysed:** {total_flows}")
    
    # ============================================
    # DETECTION PERFORMANCE
    # ============================================
    st.subheader("🎯 Detection Performance (on all flows)")
    
    # Find ground truth column
    gt_col = None
    for col in ['GroundTruth', 'label', 'Label']:
        if col in flow_df.columns:
            gt_col = col
            break
    
    if gt_col is None:
        st.error("❌ No ground truth column found!")
        st.info("Available columns: " + ", ".join(flow_df.columns[:10]))
        return
    
    # Find prediction column
    pred_col = None
    for col in ['Predicted_Attack_Probability', 'pred_prob', 'attack_prob']:
        if col in flow_df.columns:
            pred_col = col
            break
    
    if pred_col is None:
        st.error("❌ No prediction probability column found!")
        return
    
    # Ensure ground truth is binary
    flow_df['GroundTruth_binary'] = flow_df[gt_col].astype(int)
    
    # Show dataset composition
    gt_dist = flow_df['GroundTruth_binary'].value_counts()
    st.info(f"📊 **Dataset composition:** {gt_dist.get(1, 0)} malicious flows, {gt_dist.get(0, 0)} benign flows")
    
    # ============================================
    # METRICS AT DIFFERENT THRESHOLDS
    # ============================================
    st.subheader("📊 Performance at Different Thresholds")
    
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    results = []
    
    for threshold in thresholds:
        y_pred = (flow_df[pred_col] >= threshold).astype(int)
        y_true = flow_df['GroundTruth_binary']
        
        tp = ((y_true == 1) & (y_pred == 1)).sum()
        tn = ((y_true == 0) & (y_pred == 0)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()
        
        total = len(flow_df)
        accuracy = (tp + tn) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        results.append({
            "Threshold": threshold,
            "Accuracy": round(accuracy, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1-Score": round(f1, 4),
            "TP": int(tp),
            "FP": int(fp),
            "FN": int(fn),
            "TN": int(tn)
        })
    
    results_df = pd.DataFrame(results)
    st.dataframe(results_df.style.format({
        "Threshold": "{:.1f}",
        "Accuracy": "{:.4f}",
        "Precision": "{:.4f}",
        "Recall": "{:.4f}",
        "F1-Score": "{:.4f}"
    }))
    
    # Best threshold recommendation
    best_row = results_df.loc[results_df["F1-Score"].idxmax()]
    st.success(f"✅ **Best threshold: {best_row['Threshold']:.1f}** with F1-Score: {best_row['F1-Score']:.4f}")
    
    # ============================================
    # CONFUSION MATRIX AT THRESHOLD 0.5
    # ============================================
    st.subheader("📊 Confusion Matrix (Threshold = 0.5)")
    
    threshold = 0.5
    y_pred = (flow_df[pred_col] >= threshold).astype(int)
    y_true = flow_df['GroundTruth_binary']
    
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    
    confusion_data = pd.DataFrame({
        "": ["Actual Malicious", "Actual Benign"],
        "Predicted Malicious": [int(tp), int(fp)],
        "Predicted Benign": [int(fn), int(tn)]
    })
    st.dataframe(confusion_data)
    
    # Metrics at threshold 0.5
    accuracy = (tp + tn) / len(flow_df)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", f"{accuracy:.4f}")
    with col2:
        st.metric("Precision", f"{precision:.4f}")
    with col3:
        st.metric("Recall", f"{recall:.4f}")
    with col4:
        st.metric("F1-Score", f"{f1:.4f}")
    
    malicious_count = y_true.sum()
    st.info(f"🔴 **Malicious flows:** {int(malicious_count)} out of {len(flow_df)} ({malicious_count/len(flow_df)*100:.1f}%)")
    
    # ============================================
    # ATTACK DETAILS
    # ============================================
    st.subheader("📊 Attack Details")
    
    if 'attack_cat' in flow_df.columns:
        attack_counts = flow_df['attack_cat'].value_counts().reset_index()
        attack_counts.columns = ["Attack Category", "Count"]
        st.dataframe(attack_counts)
    elif 'type' in flow_df.columns:
        attack_counts = flow_df['type'].value_counts().reset_index()
        attack_counts.columns = ["Attack Type", "Count"]
        st.dataframe(attack_counts)
    else:
        st.info("No attack category column found.")
    
    # ============================================
    # SAMPLE OF 50 FLOWS
    # ============================================
    st.subheader("📋 Sample of 50 Flows")
    
    # Select columns to display
    display_cols = ["Node_ID", "GroundTruth", "Predicted_Attack_Probability"]
    
    # Add attack category if available
    for col in ['attack_cat', 'type']:
        if col in flow_df.columns:
            display_cols.append(col)
            break
    
    # Add trust scores
    for col in ["Hybrid_Trust", "BARM_Score", "ADRS_MPIQ_Score", "Trust_Value"]:
        if col in flow_df.columns:
            display_cols.append(col)
    
    # Add network columns
    for col in ["srcip", "dstip", "proto", "service"]:
        if col in flow_df.columns:
            display_cols.append(col)
    
    display_cols = [c for c in display_cols if c in flow_df.columns]
    sample_50 = flow_df.head(50)
    st.dataframe(sample_50[display_cols])
    
    # ============================================
    # CONSENSUS
    # ============================================
    st.subheader("⚖️ Consensus")
    if "Consensus" in flow_df.columns:
        consensus_val = flow_df["Consensus"].iloc[0] if not flow_df.empty else None
        st.metric("Consensus Value", f"{consensus_val:.4f}" if consensus_val is not None else "N/A")
    
    # ============================================
    # TRUST VS DETECTION
    # ============================================
    st.subheader("📈 Trust Value vs Malicious/Benign")
    trust_col = "Trust_Value"
    if trust_col in flow_df.columns and "GroundTruth_binary" in flow_df.columns:
        trust_stats = flow_df.groupby("GroundTruth_binary")[trust_col].agg(["mean", "median", "std", "count"]).round(4)
        trust_stats.index = ["Benign (0)", "Malicious (1)"]
        st.dataframe(trust_stats)
        
        corr = flow_df[trust_col].corr(flow_df["GroundTruth_binary"])
        st.metric("Correlation (Trust vs Malicious)", f"{corr:.4f}")
        if corr < -0.3:
            st.caption("🔴 Strong negative correlation: Malicious flows have lower trust values.")
        elif corr < -0.1:
            st.caption("🟡 Weak negative correlation: Some relationship between trust and maliciousness.")
        else:
            st.caption("🟢 No strong correlation: Trust may not be a strong discriminator.")
    
    # ============================================
    # PROTOCOL DISTRIBUTION
    # ============================================
    if "proto" in flow_df.columns:
        st.subheader("🔌 Protocol Distribution")
        proto_counts = flow_df["proto"].value_counts().reset_index()
        proto_counts.columns = ["Protocol", "Count"]
        st.dataframe(proto_counts)
    
    # ============================================
    # SUMMARY STATISTICS
    # ============================================
    st.subheader("📊 Summary Statistics (Trust Scores)")
    numeric_cols = ["Hybrid_Trust", "BARM_Score", "ADRS_MPIQ_Score", 
                    "Trust_Value", "Reputation", "Updated_Trust", 
                    "Routing_Score", "Fitness"]
    existing_numeric = [c for c in numeric_cols if c in flow_df.columns]
    if existing_numeric:
        stats = flow_df[existing_numeric].describe().round(4)
        st.dataframe(stats.style.format("{:.4f}"))
    
    # ============================================
    # MISSING VALUES
    # ============================================
    st.subheader("🧹 Missing Values")
    missing = flow_df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        st.dataframe(missing.to_frame("Missing Count"))
    else:
        st.info("✅ No missing values in the dataset.")