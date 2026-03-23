using UnityEngine;

public class PoseCapture : MonoBehaviour
{
    [Header("XR References")]
    public Transform head;
    public Transform leftHand;
    public Transform rightHand;

    public PoseFrame CurrentPose { get; private set; }

    void Update()
    {
        CurrentPose = CapturePose();
    }

    PoseFrame CapturePose()
    {
        PoseFrame pose = new PoseFrame();

        if (head != null)
        {
            pose.headPosition = head.position;
            pose.headRotation = head.rotation;
        }

        if (leftHand != null)
        {
            pose.leftHandPosition = leftHand.position;
            pose.leftHandRotation = leftHand.rotation;
        }

        if (rightHand != null)
        {
            pose.rightHandPosition = rightHand.position;
            pose.rightHandRotation = rightHand.rotation;
        }

        return pose;
    }
}