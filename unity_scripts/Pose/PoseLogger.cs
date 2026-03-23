using UnityEngine;
using System.IO;
using System.Text;

public class PoseLogger : MonoBehaviour
{
    public PoseCapture poseCapture;

    [Header("Logging Settings")]
    public bool enableLogging = true;
    public float logInterval = 0.1f; // seconds (10 logs/sec)

    private float timer = 0f;
    private string filePath;

    void Start()
    {
        filePath = Path.Combine(Application.persistentDataPath, "pose_log.txt");

        // Clear file at start
        File.WriteAllText(filePath, "timestamp,headPos,leftHandPos,rightHandPos\n");
    }

    void Update()
    {
        if (!enableLogging || poseCapture == null) return;

        timer += Time.deltaTime;

        if (timer >= logInterval)
        {
            timer = 0f;
            LogPose(poseCapture.CurrentPose);
        }
    }

    void LogPose(PoseFrame pose)
    {
        if (pose == null) return;

        StringBuilder sb = new StringBuilder();

        sb.Append(pose.timestamp).Append(",");
        sb.Append(pose.headPosition.ToString("F3")).Append(",");
        sb.Append(pose.leftHandPosition.ToString("F3")).Append(",");
        sb.Append(pose.rightHandPosition.ToString("F3")).Append("\n");

        File.AppendAllText(filePath, sb.ToString());
    }
}