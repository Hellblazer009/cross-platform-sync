using UnityEngine;

public class PoseCapture : MonoBehaviour
{
    public Transform headTransform;

    void Update()
    {
        Vector3 position = headTransform.position;
        Quaternion rotation = headTransform.rotation;

        Debug.Log("Head Position: " + position);
        Debug.Log("Head Rotation: " + rotation);
    }
}