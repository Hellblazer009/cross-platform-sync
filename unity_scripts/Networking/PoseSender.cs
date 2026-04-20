using System.Collections;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;

/// <summary>
/// Sends raw pose frames to the edge server at a controlled rate.
/// Also manages the clock sync protocol with the edge server.
///
/// Features vs old PoseSender:
///   - Rate-limited to sendRateHz (default 30Hz) — not every frame
///   - Sends full pose: head + left hand + right hand
///   - Delta encoding: skips send if pose moved less than positionThreshold
///   - Clock sync protocol: POST /sync_clock + POST /sync_ack every syncIntervalSec
///   - Device type and tracking mode configurable per platform
/// </summary>
public class PoseSender : MonoBehaviour
{
    [Header("Edge Server")]
    public string edgeServerUrl   = "http://localhost:8000";
    public string userId          = "user_1";

    [Header("Device")]
    [Tooltip("VR_6DOF | AR_6DOF | DOF_3 | SYNTHETIC")]
    public string deviceType      = "VR_6DOF";
    [Tooltip("6DoF | 3DoF | Synthetic")]
    public string trackingMode    = "6DoF";

    [Header("XR References")]
    public Transform headTransform;
    public Transform leftHandTransform;
    public Transform rightHandTransform;

    [Header("Rate Control")]
    [Range(5f, 90f)]
    [Tooltip("Pose transmissions per second to the edge server.")]
    public float sendRateHz       = 30f;

    [Tooltip("Skip send if head moved less than this (metres). 0 = always send.")]
    public float positionThreshold = 0.001f;  // 1 mm

    [Header("Clock Sync")]
    [Tooltip("How often to run the clock sync protocol (seconds).")]
    public float syncIntervalSec  = 5f;

    // ── Private state ──────────────────────────────────────────────────────
    private float _sendTimer;
    private float _syncTimer;
    private Vector3 _lastSentHeadPos;
    private bool  _syncInProgress;

    void Update()
    {
        _sendTimer += Time.deltaTime;
        _syncTimer += Time.deltaTime;

        if (_sendTimer >= 1f / sendRateHz)
        {
            _sendTimer = 0f;
            TrySendPose();
        }

        if (_syncTimer >= syncIntervalSec && !_syncInProgress)
        {
            _syncTimer = 0f;
            StartCoroutine(RunClockSync());
        }
    }

    // ── Pose transmission ──────────────────────────────────────────────────

    void TrySendPose()
    {
        if (headTransform == null) return;

        // Delta encoding: skip if barely moved
        if (positionThreshold > 0f &&
            Vector3.Distance(headTransform.position, _lastSentHeadPos) < positionThreshold)
            return;

        _lastSentHeadPos = headTransform.position;

        var frame = RawPoseFrame.Build(
            userId, deviceType, trackingMode,
            headTransform, leftHandTransform, rightHandTransform
        );

        StartCoroutine(PostJson("/pose", JsonUtility.ToJson(frame)));
    }

    // ── Clock sync protocol ────────────────────────────────────────────────

    IEnumerator RunClockSync()
    {
        _syncInProgress = true;

        // Step 1: send t_c1
        float t_c1   = Time.realtimeSinceStartup;
        var step1Req = new SyncClockRequest { user_id = userId, t_c1 = t_c1 };
        string step1Json = JsonUtility.ToJson(step1Req);

        float t_e = 0f;
        bool  step1Ok = false;

        using (var req = BuildPostRequest("/sync_clock", step1Json))
        {
            yield return req.SendWebRequest();
            if (req.result == UnityWebRequest.Result.Success)
            {
                var resp = JsonUtility.FromJson<SyncClockResponse>(req.downloadHandler.text);
                t_e     = resp.t_e;
                step1Ok = true;
            }
        }

        if (!step1Ok) { _syncInProgress = false; yield break; }

        // Step 2: send t_c2 (time we received the response)
        float t_c2   = Time.realtimeSinceStartup;
        var step2Req = new SyncAckRequest
        {
            user_id = userId,
            t_c1    = t_c1,
            t_e     = t_e,
            t_c2    = t_c2,
        };

        yield return PostJson("/sync_ack", JsonUtility.ToJson(step2Req));

        _syncInProgress = false;
    }

    // ── HTTP helpers ───────────────────────────────────────────────────────

    IEnumerator PostJson(string path, string json)
    {
        using var req = BuildPostRequest(path, json);
        yield return req.SendWebRequest();
        if (req.result != UnityWebRequest.Result.Success)
            Debug.LogWarning($"[PoseSender] {path} failed: {req.error}");
    }

    UnityWebRequest BuildPostRequest(string path, string json)
    {
        var req = new UnityWebRequest(edgeServerUrl + path, "POST");
        var body = Encoding.UTF8.GetBytes(json);
        req.uploadHandler   = new UploadHandlerRaw(body);
        req.downloadHandler = new DownloadHandlerBuffer();
        req.SetRequestHeader("Content-Type", "application/json");
        return req;
    }

    // ── Internal serialisable types for clock sync ─────────────────────────

    [System.Serializable] private class SyncClockRequest  { public string user_id; public float t_c1; }
    [System.Serializable] private class SyncClockResponse { public float t_e; }
    [System.Serializable] private class SyncAckRequest    { public string user_id; public float t_c1; public float t_e; public float t_c2; }
}
