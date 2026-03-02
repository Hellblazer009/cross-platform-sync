using UnityEngine;
using NativeWebSocket;
using System;

public class EdgeListener : MonoBehaviour
{
    WebSocket websocket;

    async void Start()
    {
        websocket = new WebSocket("ws://localhost:8765");

        websocket.OnMessage += (bytes) =>
        {
            string json = System.Text.Encoding.UTF8.GetString(bytes);
            InterpretedPoseFrame pose =
                JsonUtility.FromJson<InterpretedPoseFrame>(json);

            PublishToNormcore(pose);
        };

        await websocket.Connect();
    }

    void PublishToNormcore(InterpretedPoseFrame pose)
    {
        // Set Normcore RealtimeModel values here
    }
}