using UnityEngine;
using System.IO;
using System.Text;

/// <summary>
/// Logs the local avatar pose to a CSV file for offline TCN training.
///
/// CSV format (one row per sample):
///   timestamp, headPos, headRot, leftHandPos, leftHandRot, rightHandPos, rightHandRot
///
/// Positions  : Unity Vector3 ToString "F4" → "(x, y, z)"
/// Rotations  : Unity Quaternion [x,y,z,w] manually formatted → "(x, y, z, w)"
///              (matching the convention used by RawPoseFrame / edge server)
///
/// The training script (train_tcn.py) parses these parenthesised groups in order.
/// </summary>
public class PoseLogger : MonoBehaviour
{
    public PoseCapture poseCapture;

    [Header("Logging Settings")]
    public bool  enableLogging = true;
    [Tooltip("How many pose samples per second to write (default 10 Hz).")]
    public float logRateHz     = 10f;

    [Header("File")]
    [Tooltip("File name inside Application.persistentDataPath.")]
    public string fileName = "pose_log.csv";

    // ── Private ─────────────────────────────────────────────────────────────
    private float  _timer;
    private string _filePath;

    static readonly string k_Header =
        "timestamp," +
        "headPos,headRot," +
        "leftHandPos,leftHandRot," +
        "rightHandPos,rightHandRot\n";

    void Start()
    {
        _filePath = Path.Combine(Application.persistentDataPath, fileName);
        File.WriteAllText(_filePath, k_Header);
        Debug.Log($"[PoseLogger] Writing to: {_filePath}");
    }

    void Update()
    {
        if (!enableLogging || poseCapture == null) return;

        _timer += Time.deltaTime;
        if (_timer < 1f / logRateHz) return;

        _timer = 0f;
        LogPose(poseCapture.CurrentPose);
    }

    void LogPose(PoseFrame pose)
    {
        if (pose == null) return;

        var sb = new StringBuilder(256);

        sb.Append(pose.timestamp.ToString("F4")).Append(',');
        AppendVec3(sb, pose.headPosition);       sb.Append(',');
        AppendQuat(sb, pose.headRotation);       sb.Append(',');
        AppendVec3(sb, pose.leftHandPosition);   sb.Append(',');
        AppendQuat(sb, pose.leftHandRotation);   sb.Append(',');
        AppendVec3(sb, pose.rightHandPosition);  sb.Append(',');
        AppendQuat(sb, pose.rightHandRotation);  sb.Append('\n');

        File.AppendAllText(_filePath, sb.ToString());
    }

    // ── Format helpers ────────────────────────────────────────────────────

    static void AppendVec3(StringBuilder sb, Vector3 v)
    {
        sb.Append('(')
          .Append(v.x.ToString("F4")).Append(", ")
          .Append(v.y.ToString("F4")).Append(", ")
          .Append(v.z.ToString("F4"))
          .Append(')');
    }

    /// <summary>
    /// Write quaternion as (x, y, z, w) — Unity convention.
    /// The training script converts to [w, x, y, z] internally.
    /// </summary>
    static void AppendQuat(StringBuilder sb, Quaternion q)
    {
        sb.Append('(')
          .Append(q.x.ToString("F4")).Append(", ")
          .Append(q.y.ToString("F4")).Append(", ")
          .Append(q.z.ToString("F4")).Append(", ")
          .Append(q.w.ToString("F4"))
          .Append(')');
    }
}