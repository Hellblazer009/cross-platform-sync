[System.Serializable]
public class RawPoseFrame {
    public string user_id;
    public string device_type;
    public string tracking_mode;
    public float timestamp;
    public Vector3 position;
    public Quaternion rotation;
    public float confidence;
}