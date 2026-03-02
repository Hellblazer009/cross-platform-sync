[System.Serializable]
public class InterpretedPoseFrame {
    public string user_id;
    public string interpretation;
    public Vector3 position;
    public Quaternion rotation;
    public float confidence;
    public float server_timestamp;
}