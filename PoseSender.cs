using UnityEngine;
using UnityEngine.Networking;
using System.Collections;
using System.Text;

public class PoseSender : MonoBehaviour
{
    public string edgeUrl = "http://localhost:8000/pose";
    public string userId = "user_1";
    public Transform headTransform;

    void Update()
    {
        StartCoroutine(SendPose());
    }

    IEnumerator SendPose()
    {
        RawPoseFrame pose = new RawPoseFrame {
            user_id = userId,
            device_type = "VR",
            tracking_mode = "6DoF",
            timestamp = Time.time,
            position = headTransform.position,
            rotation = headTransform.rotation,
            confidence = 1.0f
        };

        string json = JsonUtility.ToJson(pose);

        UnityWebRequest request = new UnityWebRequest(edgeUrl, "POST");
        byte[] body = Encoding.UTF8.GetBytes(json);
        request.uploadHandler = new UploadHandlerRaw(body);
        request.downloadHandler = new DownloadHandlerBuffer();
        request.SetRequestHeader("Content-Type", "application/json");

        yield return request.SendWebRequest();
    }
}