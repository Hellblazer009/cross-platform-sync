using UnityEngine;

/// <summary>
/// Data contract sent from each Unity client to the edge server via POST /pose.
/// Matches the Python RawPoseFrame Pydantic model exactly.
///
/// Device type strings:
///   "VR_6DOF"   — Meta Quest (full 6DoF)
///   "AR_6DOF"   — Xreal Beam Pro + Air glasses (6DoF SLAM)
///   "DOF_3"     — rotation-only devices
///   "SYNTHETIC" — desktop / simulated
///
/// Tracking mode strings:
///   "6DoF" | "3DoF" | "Synthetic"
///
/// Quaternion convention: Unity [x, y, z, w] — edge server converts internally.
/// </summary>
[System.Serializable]
public class RawPoseFrame
{
    public string user_id;
    public string device_type;       // "VR_6DOF" | "AR_6DOF" | "DOF_3" | "SYNTHETIC"
    public string tracking_mode;     // "6DoF" | "3DoF" | "Synthetic"
    public float  timestamp;         // Time.realtimeSinceStartup (client monotonic)

    // Head pose
    public float[] head_position;    // [x, y, z]
    public float[] head_rotation;    // [x, y, z, w]

    // Left hand
    public float[] left_hand_position;
    public float[] left_hand_rotation;

    // Right hand
    public float[] right_hand_position;
    public float[] right_hand_rotation;

    public float confidence;

    // ── Factory helpers ────────────────────────────────────────────────────

    public static float[] Vec3ToArray(Vector3 v)
        => new float[] { v.x, v.y, v.z };

    public static float[] QuatToArray(Quaternion q)
        => new float[] { q.x, q.y, q.z, q.w };   // Unity convention

    /// <summary>
    /// Build a RawPoseFrame from live XR transforms.
    /// Pass null for any hand that is not tracked.
    /// </summary>
    public static RawPoseFrame Build(
        string    userId,
        string    deviceType,
        string    trackingMode,
        Transform headTransform,
        Transform leftHandTransform,
        Transform rightHandTransform,
        float     confidence = 1.0f)
    {
        var frame = new RawPoseFrame
        {
            user_id       = userId,
            device_type   = deviceType,
            tracking_mode = trackingMode,
            timestamp     = Time.realtimeSinceStartup,
            confidence    = confidence,
        };

        if (headTransform != null)
        {
            frame.head_position = Vec3ToArray(headTransform.position);
            frame.head_rotation = QuatToArray(headTransform.rotation);
        }
        else
        {
            frame.head_position = new float[] { 0, 0, 0 };
            frame.head_rotation = new float[] { 0, 0, 0, 1 };
        }

        if (leftHandTransform != null)
        {
            frame.left_hand_position = Vec3ToArray(leftHandTransform.position);
            frame.left_hand_rotation = QuatToArray(leftHandTransform.rotation);
        }
        else
        {
            frame.left_hand_position = new float[] { 0, 0, 0 };
            frame.left_hand_rotation = new float[] { 0, 0, 0, 1 };
        }

        if (rightHandTransform != null)
        {
            frame.right_hand_position = Vec3ToArray(rightHandTransform.position);
            frame.right_hand_rotation = QuatToArray(rightHandTransform.rotation);
        }
        else
        {
            frame.right_hand_position = new float[] { 0, 0, 0 };
            frame.right_hand_rotation = new float[] { 0, 0, 0, 1 };
        }

        return frame;
    }
}
