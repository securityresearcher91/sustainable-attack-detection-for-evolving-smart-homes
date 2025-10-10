import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
from scipy.spatial.distance import cdist
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

sim_data = 'datasets/camera_simulator.csv'
#real_data = 'datasets/tplink_camera.csv'
real_data = 'datasets/dlink_camera.csv'

def compute_wasserstein_distance(df1, df2):
    print("\nComputing Wasserstein Distance")
    common_cols = df1.columns.intersection(df2.columns)
    df1_num = df1.select_dtypes(include='number').dropna()
    df2_num = df2.select_dtypes(include='number').dropna()
    distances = {}
    for col in common_cols:
        series1 = df1[col].dropna()
        series2 = df2[col].dropna()
        if pd.api.types.is_numeric_dtype(series1) and pd.api.types.is_numeric_dtype(series2):
            dist = wasserstein_distance(series1, series2)
            distances[col] = dist
    for col, dist in sorted(distances.items(), key=lambda x: -x[1]):
        print(f"{col}: {dist}")

def compute_jsd(p, q, bins=100):
    p = np.asarray(p)
    q = np.asarray(q)
    p_min, p_max = np.min(p), np.max(p)
    q_min, q_max = np.min(q), np.max(q)
    global_min = min(p_min, q_min)
    global_max = max(p_max, q_max)
    if global_max - global_min == 0:
        return 0.0
    p_norm = (p - global_min) / (global_max - global_min)
    q_norm = (q - global_min) / (global_max - global_min)
    p_hist, _ = np.histogram(p_norm, bins=bins, range=(0, 1), density=True)
    q_hist, _ = np.histogram(q_norm, bins=bins, range=(0, 1), density=True)
    epsilon = 1e-12
    p_hist = np.where(p_hist == 0, epsilon, p_hist)
    q_hist = np.where(q_hist == 0, epsilon, q_hist)
    p_hist /= p_hist.sum()
    q_hist /= q_hist.sum()
    m = 0.5 * (p_hist + q_hist)
    js_div = 0.5 * (entropy(p_hist, m, base=2) + entropy(q_hist, m, base=2))
    return np.sqrt(js_div)

def compute_jensonshannon_distance(df1, df2):
    print("\nComputing Jenson Shannon Distance")
    feature_distances = {}
    common_features = set(df1.columns) & set(df2.columns)
    for feature in sorted(common_features):
        sim_vals = df1[feature].dropna()
        real_vals = df2[feature].dropna()
        if len(sim_vals) > 0 and len(real_vals) > 0:
            try:
                jsd = compute_jsd(sim_vals, real_vals)
                feature_distances[feature] = jsd
            except Exception as e:
                print(f"Error computing JSD for {feature}: {e}")
    for feature, distance in sorted(feature_distances.items(), key=lambda x: -x[1]):
        print(f"{feature}: {distance:.4f}")

def binarize_feature(series, threshold='median'):
    if threshold == 'median':
        t = series.median()
    elif threshold == 'mean':
        t = series.mean()
    else:
        t = float(threshold)
    return (series > t).astype(int)

def compute_tanimoto_distance(df1, df2):
    print("\nComputing Tanimoto Distance")
    feature_distances = {}
    common_features = sorted(set(df1.columns) & set(df2.columns))
    for feature in common_features:
        s1 = df1[feature].dropna()
        s2 = df2[feature].dropna()
        if len(s1) > 0 and len(s2) > 0:
            try:
                n = min(len(s1), len(s2))
                s1_bin = binarize_feature(s1.iloc[:n])
                s2_bin = binarize_feature(s2.iloc[:n])
                s1_bin = s1_bin.values.reshape(1, -1)
                s2_bin = s2_bin.values.reshape(1, -1)
                tanimoto_sim = 1 - cdist(s1_bin, s2_bin, metric='jaccard')[0][0]
                tanimoto_dist = 1 - tanimoto_sim
                feature_distances[feature] = tanimoto_dist
            except Exception as e:
                print(f"Error computing Tanimoto for {feature}: {e}")
    sorted_features = sorted(feature_distances.items(), key=lambda x: -x[1])
    for feature, distance in sorted_features:
        print(f"{feature}: {distance:.4f}")

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pandas as pd

def compute_discriminator_fpr(sim_df, real_df):
    print("\nComputing Discriminator Evaluation using Isolation Forest")

    # Use only common numeric columns
    common_cols = real_df.columns.intersection(sim_df.columns)
    sim = sim_df[common_cols].select_dtypes(include='number').dropna()
    real = real_df[common_cols].select_dtypes(include='number').dropna()
    sim = sim[real.columns.intersection(sim.columns)]
    real = real[sim.columns]

    # Standardise features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(sim)
    X_test = scaler.transform(real)

    # Train on synthetic
    model = IsolationForest(contamination='auto', random_state=42)
    model.fit(X_train)
    
    real_preds = model.predict(X_test)  # 1: normal, -1: anomaly

    # Evaluate FPR: how many real samples are falsely classified as anomalies
    y_true = np.ones(len(real_preds))        # all real data should be normal (1)
    y_pred = (real_preds == 1).astype(int)   # model predicts 1 for normal
    print(y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    print(f"False Positive Rate (FPR): {fpr:.4f} — Higher indicates greater similarity")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

def compute_rf_discriminator_fpr_trainB_testA(sim_df, real_df, random_state=42):
    print("\nComputing Discriminator Evaluation using Random Forest (Train on B, Test on A′)")

    # Intersect numeric columns
    common_cols = real_df.columns.intersection(sim_df.columns)
    real = real_df[common_cols].select_dtypes(include='number').dropna()
    sim = sim_df[common_cols].select_dtypes(include='number').dropna()
    real = real[real.columns.intersection(sim.columns)]
    sim = sim[real.columns]

    # Assign labels
    sim['label'] = 0  # Synthetic = class 0
    real['label'] = 1  # Real = class 1

    # Train only on synthetic (B), test only on real (A′)
    X_train = sim.drop(columns='label')
    y_train = sim['label']
    X_test = real.drop(columns='label')
    y_test = real['label']

    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest
    clf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    clf.fit(X_train_scaled, y_train)

    # Predict on real data
    y_pred = clf.predict(X_test_scaled)

    # Evaluate FPR: proportion of real samples predicted as synthetic (i.e. false negatives for real == 0s)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) != 0 else 0

    print(f"False Positive Rate (FPR): {fpr:.4f} — Higher indicates greater similarity")

    return fpr


def drop_irrelevant_features(df):
    irrelevant_features = [
        # Identifiers
        'src_ip', 'dst_ip', 'src_port', 'dst_port', 'timestamp',

        # Rare or noisy TCP flags
        'cwr_flag_count', 'ece_flag_cnt', 'urg_flag_cnt',
        'fwd_urg_flags', 'bwd_urg_flags',

        # Constant or mostly-zero temporal features
        'active_max', 'active_min', 'active_mean', 'active_std',
        'idle_max', 'idle_min', 'idle_mean', 'idle_std',

        # Redundant metrics
        'fwd_seg_size_min',

        # Redundant subflow aggregates
        'subflow_fwd_byts', 'subflow_bwd_byts',
        'subflow_fwd_pkts', 'subflow_bwd_pkts'
    ]

    # Only drop columns that exist in the DataFrame
    existing = [col for col in irrelevant_features if col in df.columns]
    return df.drop(columns=existing)

def select_flow_packet_features(df):
    """
    Selects only the flow and packet characteristics:
    - Flow duration
    - Flow size (bytes per second, packets per second, total bytes)
    - Packet timing
    - Packet sizes
    - Number of packets
    """
    # Define relevant feature substrings
    keywords = [
        'flow_duration',            # Flow duration
        'flow_byts_s', 'flow_pkts_s',  # Flow sizes
        'pkt_len', 'pkt_size',         # Packet sizes
        'iat',                         # Inter-arrival times
        'tot_bwd_pkts', 'tot_fwd_pkts', 'totlen_bwd_pkts', 'totlen_fwd_pkts'  # Packet counts/sizes
    ]

    # Match columns containing any keyword
    selected_cols = [col for col in df.columns if any(key in col.lower() for key in keywords)]

    return df[selected_cols]

def main():
    df1 = pd.read_csv(sim_data)
    sim_df = select_flow_packet_features(df1)

    df2 = pd.read_csv(real_data)
    real_df = select_flow_packet_features(df2)

    compute_wasserstein_distance(sim_df, real_df)
    compute_jensonshannon_distance(sim_df, real_df)
    compute_tanimoto_distance(sim_df, real_df)
    compute_discriminator_fpr(sim_df, real_df)
    compute_rf_discriminator_fpr_trainB_testA(sim_df, real_df)

if __name__=="__main__":
    main()

