import numpy as np
import pandas as pd

def generate_network_traffic_data(n_samples=1000, anomaly_ratio=0.5, label_noise=0.05):
    n_anomalous = int(n_samples * anomaly_ratio)
    n_normal = n_samples - n_anomalous
    
    # Normal traffic
    normal_packets = np.random.normal(loc=100, scale=10, size=n_normal)
    normal_bytes = normal_packets * np.random.normal(loc=10, scale=2, size=n_normal)
    normal_drop_rate = np.random.normal(loc=0.01, scale=0.005, size=n_normal)
#     normal_protocol = np.random.choice([1], size=n_normal) # TCP only
    
    # Anomalous traffic
    anomaly_packets = np.random.normal(loc=300, scale=40, size=n_anomalous)
    anomaly_bytes = anomaly_packets * np.random.normal(loc=15, scale=5, size=n_anomalous)
    anomaly_drop_rate = np.random.normal(loc=0.1, scale=0.03, size=n_anomalous)
#     anomaly_protocol = np.random.choice([2,3], size=n_anomalous) # sudden protocol change (e.g. UDP, ICMP)
    
    X = np.vstack([
        np.column_stack((normal_packets, normal_bytes, normal_drop_rate)),
        np.column_stack((anomaly_packets, anomaly_bytes, anomaly_drop_rate))
    ])
    
    y = np.array([-1]*n_normal + [1]*n_anomalous)
    
    return pd.DataFrame(X, columns=["packets_per_sec", "bytes_per_sec", "packet_drop_rate"]), y

def generate_network_traffic_data_2(n_samples=1000, anomaly_ratio=0.05, label_noise=0.05):
    n_anomalous = int(n_samples * anomaly_ratio)
    n_normal = n_samples - n_anomalous
    
    # Normal traffic
    normal_packets  = np.random.normal(120, 40, n_normal)
    normal_bytes = np.random.lognormal(mean=6, sigma=0.6, size=n_normal)
    normal_drop_rate = np.random.normal(loc=0.01, scale=0.005, size=n_normal)
#     normal_protocol = np.random.choice([1], size=n_normal) # TCP only
    
    # Anomalous traffic
    anomaly_packets = np.random.normal(160, 60, n_anomalous)
    anomaly_bytes = np.random.lognormal(mean=6.2, sigma=0.9, size=n_anomalous)
    anomaly_drop_rate = np.random.normal(loc=0.1, scale=0.03, size=n_anomalous)
#     anomaly_protocol = np.random.choice([2,3], size=n_anomalous) # sudden protocol change (e.g. UDP, ICMP)
    
    X = np.vstack([
        np.column_stack((normal_packets, normal_bytes, normal_drop_rate)),
        np.column_stack((anomaly_packets, anomaly_bytes, anomaly_drop_rate))
    ])
    
    y = np.array([-1]*n_normal + [1]*n_anomalous)
    flip_idx = np.random.choice(len(y), size=int(label_noise*len(y)), replace=False)
    y[flip_idx] *= -1
    
    return pd.DataFrame(X, columns=["packets_per_sec", "bytes_per_sec", "packet_drop_rate"]), y