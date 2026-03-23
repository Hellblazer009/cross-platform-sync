using UnityEngine;

[System.Serializable]
public class PoseFrame
{
    public Vector3 headPosition;
    public Quaternion headRotation;

    public Vector3 leftHandPosition;
    public Quaternion leftHandRotation;

    public Vector3 rightHandPosition;
    public Quaternion rightHandRotation;

    public float timestamp;

    public PoseFrame()
    {
        timestamp = Time.time;
    }
}