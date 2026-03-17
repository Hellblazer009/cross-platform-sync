using UnityEngine;
using UnityEngine.XR;

public class XRHeadPoseReader : MonoBehaviour
{
    InputDevice headDevice;

    void Start()
    {
        headDevice = InputDevices.GetDeviceAtXRNode(XRNode.Head);
    }

    void Update()
    {
        if (!headDevice.isValid)
        {
            headDevice = InputDevices.GetDeviceAtXRNode(XRNode.Head);
            return;
        }

        Vector3 position;
        Quaternion rotation;

        if (headDevice.TryGetFeatureValue(CommonUsages.devicePosition, out position) &&
            headDevice.TryGetFeatureValue(CommonUsages.deviceRotation, out rotation))
        {
            Debug.Log("Head Position: " + position);
            Debug.Log("Head Rotation: " + rotation);
        }
    }
}