using UnityEngine;

/// <summary>
/// Data contract received from the edge server via WebSocket /ws.
/// Matches the Python _build_payload() dict exactly.
///
/// prediction_source values: "physics" | "blended" | "residual" | "baseline"
/// blend_alpha: 0 = physics only, 1 = full TCN residual correction
/// uncertainty: per-dimension variance [Δp(3), Δφ(3)] — lower = more confident
/// </summary>
[System.Serializable]
public class InterpretedPoseFrame
{
    public string user_id;
    public string interpretation;         // "FullBody" | "HeadOnly" | "Proxy"
    public float  corrected_timestamp;    // client timestamp in edge clock domain
    public float  edge_timestamp;         // edge server reception time

    // Head pose — Unity [x,y,z,w] convention (edge server converts back)
    public float[] head_position;
    public float[] head_rotation;

    // Left hand
    public float[] left_hand_position;
    public float[] left_hand_rotation;

    // Right hand
    public float[] right_hand_position;
    public float[] right_hand_rotation;

    // Quality metrics
    public float   confidence;
    public float[] uncertainty;           // length 6: [σ²_p(3), σ²_φ(3)]
    public float   blend_alpha;
    public string  prediction_source;
    public float   mahalanobis_distance;

    // ── Conversion helpers ─────────────────────────────────────────────────

    public Vector3 HeadPosition =>
        head_position != null && head_position.Length >= 3
            ? new Vector3(head_position[0], head_position[1], head_position[2])
            : Vector3.zero;

    public Quaternion HeadRotation =>
        head_rotation != null && head_rotation.Length >= 4
            ? new Quaternion(head_rotation[0], head_rotation[1],
                             head_rotation[2], head_rotation[3])
            : Quaternion.identity;

    public Vector3 LeftHandPosition =>
        left_hand_position != null && left_hand_position.Length >= 3
            ? new Vector3(left_hand_position[0], left_hand_position[1], left_hand_position[2])
            : Vector3.zero;

    public Quaternion LeftHandRotation =>
        left_hand_rotation != null && left_hand_rotation.Length >= 4
            ? new Quaternion(left_hand_rotation[0], left_hand_rotation[1],
                             left_hand_rotation[2], left_hand_rotation[3])
            : Quaternion.identity;

    public Vector3 RightHandPosition =>
        right_hand_position != null && right_hand_position.Length >= 3
            ? new Vector3(right_hand_position[0], right_hand_position[1], right_hand_position[2])
            : Vector3.zero;

    public Quaternion RightHandRotation =>
        right_hand_rotation != null && right_hand_rotation.Length >= 4
            ? new Quaternion(right_hand_rotation[0], right_hand_rotation[1],
                             right_hand_rotation[2], right_hand_rotation[3])
            : Quaternion.identity;

    /// <summary>
    /// Convert to a PoseFrame for use with AvatarController.ApplyPose().
    /// </summary>
    public PoseFrame ToPoseFrame()
    {
        return new PoseFrame
        {
            headPosition      = HeadPosition,
            headRotation      = HeadRotation,
            leftHandPosition  = LeftHandPosition,
            leftHandRotation  = LeftHandRotation,
            rightHandPosition = RightHandPosition,
            rightHandRotation = RightHandRotation,
            timestamp         = edge_timestamp,
        };
    }

    /// <summary>
    /// Mean positional uncertainty (metres²) — useful for fading
    /// low-confidence remote avatars.
    /// </summary>
    public float MeanPositionalUncertainty =>
        uncertainty != null && uncertainty.Length >= 3
            ? (uncertainty[0] + uncertainty[1] + uncertainty[2]) / 3f
            : 0f;
}
