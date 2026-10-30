import numpy as np
import pandas as pd

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts domain-specific network traffic flow features designed
    to capture behavioral patterns that generalize across unseen network profiles.
    """
    df = df.copy()
    eps = 1e-5

    # 1. Port Classifications (IANA Port Ranges)
    df["orig_port_well_known"] = (df["id.orig_p"] < 1024).astype(np.float32)
    df["orig_port_registered"] = ((df["id.orig_p"] >= 1024) & (df["id.orig_p"] < 49152)).astype(np.float32)
    df["orig_port_dynamic"] = (df["id.orig_p"] >= 49152).astype(np.float32)

    df["resp_port_well_known"] = (df["id.resp_p"] < 1024).astype(np.float32)
    df["resp_port_registered"] = ((df["id.resp_p"] >= 1024) & (df["id.resp_p"] < 49152)).astype(np.float32)
    df["resp_port_dynamic"] = (df["id.resp_p"] >= 49152).astype(np.float32)

    # Standard service indicators based on destination port
    df["is_web_port"] = df["id.resp_p"].isin([80, 443, 8080, 8443]).astype(np.float32)
    df["is_dns_port"] = df["id.resp_p"].isin([53, 5353]).astype(np.float32)
    df["is_ftp_port"] = df["id.resp_p"].isin([20, 21]).astype(np.float32)
    df["is_ssh_telnet_port"] = df["id.resp_p"].isin([22, 23]).astype(np.float32)
    df["is_iot_port"] = df["id.resp_p"].isin([1883, 8883]).astype(np.float32)

    # 2. Total Packets and Packet Flow Asymmetry
    df["pkts_tot"] = df["fwd_pkts_tot"] + df["bwd_pkts_tot"]
    df["pkts_ratio"] = (df["fwd_pkts_tot"] + eps) / (df["bwd_pkts_tot"] + eps)
    df["data_pkts_tot"] = df["fwd_data_pkts_tot"] + df["bwd_data_pkts_tot"]
    df["data_pkts_ratio"] = (df["fwd_data_pkts_tot"] + eps) / (df["pkts_tot"] + eps)

    # 3. Payload Totals and Asymmetry
    df["payload_tot"] = df["fwd_pkts_payload.tot"] + df["bwd_pkts_payload.tot"]
    df["payload_ratio"] = (df["fwd_pkts_payload.tot"] + eps) / (df["bwd_pkts_payload.tot"] + eps)
    df["bytes_per_fwd_pkt"] = (df["fwd_pkts_payload.tot"] + eps) / (df["fwd_pkts_tot"] + eps)
    df["bytes_per_bwd_pkt"] = (df["bwd_pkts_payload.tot"] + eps) / (df["bwd_pkts_tot"] + eps)
    df["bytes_per_flow_pkt"] = (df["payload_tot"] + eps) / (df["pkts_tot"] + eps)

    # 4. Header vs Payload Dynamics (critical for identifying DoS/Probing vs Benign flows)
    df["header_size_tot"] = df["fwd_header_size_tot"] + df["bwd_header_size_tot"]
    df["header_to_payload_ratio"] = (df["header_size_tot"] + eps) / (df["payload_tot"] + eps)
    df["fwd_header_ratio"] = (df["fwd_header_size_tot"] + eps) / (df["header_size_tot"] + eps)

    # 5. TCP Flag Patterns
    df["total_flags"] = (
        df["flow_FIN_flag_count"]
        + df["flow_SYN_flag_count"]
        + df["flow_RST_flag_count"]
        + df["fwd_PSH_flag_count"]
        + df["bwd_PSH_flag_count"]
        + df["flow_ACK_flag_count"]
        + df["flow_CWR_flag_count"]
        + df["flow_ECE_flag_count"]
    )
    df["syn_ack_ratio"] = (df["flow_SYN_flag_count"] + eps) / (df["flow_ACK_flag_count"] + eps)
    df["rst_to_pkts_ratio"] = (df["flow_RST_flag_count"] + eps) / (df["pkts_tot"] + eps)
    df["fin_to_pkts_ratio"] = (df["flow_FIN_flag_count"] + eps) / (df["pkts_tot"] + eps)
    df["psh_flag_tot"] = df["fwd_PSH_flag_count"] + df["bwd_PSH_flag_count"]

    # 6. Window Size Dynamics
    df["window_diff"] = df["fwd_init_window_size"] - df["bwd_init_window_size"]
    df["window_ratio"] = (df["fwd_init_window_size"] + eps) / (df["bwd_init_window_size"] + eps)
    df["fwd_window_change"] = df["fwd_last_window_size"] - df["fwd_init_window_size"]

    # 7. Inter-Arrival Time (IAT) and Active/Idle Dynamics
    df["flow_iat_spread"] = df["flow_iat.max"] - df["flow_iat.min"]
    df["fwd_iat_spread"] = df["fwd_iat.max"] - df["fwd_iat.min"]
    df["bwd_iat_spread"] = df["bwd_iat.max"] - df["bwd_iat.min"]
    df["fwd_bwd_iat_ratio"] = (df["fwd_iat.avg"] + eps) / (df["bwd_iat.avg"] + eps)
    df["active_to_idle_ratio"] = (df["active.tot"] + eps) / (df["idle.tot"] + eps)

    # 8. Log1p Transformations for highly skewed variables
    skewed_cols = [
        "flow_duration",
        "flow_pkts_per_sec",
        "payload_bytes_per_second",
        "payload_tot",
        "pkts_tot",
        "header_size_tot",
        "fwd_iat.tot",
        "bwd_iat.tot",
        "flow_iat.tot",
    ]
    for col in skewed_cols:
        if col in df.columns:
            df[f"{col}_log1p"] = np.log1p(np.clip(df[col], a_min=0, a_max=None))

    # Replace any accidental infinites or NaNs
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    return df
