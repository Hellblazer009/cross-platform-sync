using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using NativeWebSocket;

/// <summary>
/// Connects persistently to the edge server WebSocket endpoint (ws://host:port/ws)
/// and drives remote avatar controllers from InterpretedPoseFrame messages.
///
/// Setup:
///   1. Install NativeWebSocket via UPM:
///      Add to Packages/manifest.json:
///        "com.endel.nativewebsocket": "https://github.com/endel/NativeWebSocket.git#upm"
///   2. Assign edgeServerWsUrl  (e.g. ws://192.168.1.10:8000)
///   3. Assign localUserId      — frames with this user_id are ignored (own echo)
///   4. Populate remoteAvatars  — Inspector list pairing user_id strings to AvatarControllers
///
/// The edge server pushes InterpretedPoseFrame JSON on /ws for every relayed frame.
/// Each frame is routed to the matching AvatarController via its user_id field.
///
/// Reconnection: exponential back-off (1s … 32s) with jitter.
/// </summary>
public class EdgePoseReceiver : MonoBehaviour
{
    // ── Inspector ──────────────────────────────────────────────────────────

    [Header("Edge Server")]
    [Tooltip("Full WebSocket URL, e.g. ws://192.168.1.10:8000")]
    public string edgeServerWsUrl = "ws://localhost:8000";

    [Tooltip("This client's own user_id — frames with matching id are ignored.")]
    public string localUserId = "user_1";

    [Header("Remote Avatar Routing")]
    [Tooltip("Map each remote user_id to the AvatarController that should receive their pose.")]
    public List<UserAvatarBinding> remoteAvatars = new List<UserAvatarBinding>();

    [Header("Reconnection")]
    [Range(0.5f, 10f)]
    [Tooltip("Initial wait before first reconnect attempt (seconds).")]
    public float reconnectBaseDelaySec = 1f;

    [Range(1f, 60f)]
    [Tooltip("Maximum reconnect back-off ceiling (seconds).")]
    public float reconnectMaxDelaySec  = 32f;

    [Header("Diagnostics")]
    [Tooltip("Log every received frame to the console (verbose — disable in production).")]
    public bool verboseLogging = false;

    // ── Public read-only state ─────────────────────────────────────────────

    /// <summary>Current WebSocket connection state.</summary>
    public WebSocketState ConnectionState =>
        _socket != null ? _socket.State : WebSocketState.Closed;

    /// <summary>Total frames received since last connect.</summary>
    public int FramesReceived { get; private set; }

    // ── Private ────────────────────────────────────────────────────────────

    private WebSocket _socket;
    private Dictionary<string, AvatarController> _avatarMap;
    private bool _applicationQuitting;
    private Coroutine _reconnectRoutine;
    private int _reconnectAttempt;

    // ── Unity lifecycle ────────────────────────────────────────────────────

    void Awake()
    {
        // Build lookup dictionary from Inspector list
        _avatarMap = new Dictionary<string, AvatarController>(StringComparer.Ordinal);
        foreach (var binding in remoteAvatars)
        {
            if (!string.IsNullOrEmpty(binding.userId) && binding.avatarController != null)
                _avatarMap[binding.userId] = binding.avatarController;
        }
    }

    void Start()
    {
        Connect();
    }

    void Update()
    {
        // NativeWebSocket requires DispatchMessageQueue() each frame so that
        // OnMessage callbacks fire on the main thread (required for Unity API).
#if !UNITY_WEBGL || UNITY_EDITOR
        _socket?.DispatchMessageQueue();
#endif
    }

    void OnApplicationQuit()
    {
        _applicationQuitting = true;
        CloseSocket();
    }

    void OnDestroy()
    {
        _applicationQuitting = true;
        CloseSocket();
    }

    // ── Connection management ──────────────────────────────────────────────

    /// <summary>
    /// Register a remote avatar at runtime (e.g. when a new player joins).
    /// Can be called from AvatarSpawner after Realtime.Instantiate.
    /// </summary>
    public void RegisterRemoteAvatar(string userId, AvatarController controller)
    {
        if (string.IsNullOrEmpty(userId) || controller == null) return;
        _avatarMap[userId] = controller;
        Debug.Log($"[EdgePoseReceiver] Registered avatar for '{userId}'.");
    }

    /// <summary>Remove a remote avatar binding (e.g. on disconnect).</summary>
    public void UnregisterRemoteAvatar(string userId)
    {
        _avatarMap.Remove(userId);
    }

    // ── Internal helpers ───────────────────────────────────────────────────

    void Connect()
    {
        if (_applicationQuitting) return;

        string wsUrl = edgeServerWsUrl.TrimEnd('/') + "/ws";
        _socket = new WebSocket(wsUrl);

        _socket.OnOpen    += OnOpen;
        _socket.OnMessage += OnMessage;
        _socket.OnError   += OnError;
        _socket.OnClose   += OnClose;

        FramesReceived = 0;
        Debug.Log($"[EdgePoseReceiver] Connecting to {wsUrl} …");

#pragma warning disable CS4014  // awaitable not awaited — intentional fire-and-forget
        _socket.Connect();
#pragma warning restore CS4014
    }

    void CloseSocket()
    {
        if (_reconnectRoutine != null)
        {
            StopCoroutine(_reconnectRoutine);
            _reconnectRoutine = null;
        }

        if (_socket != null && _socket.State == WebSocketState.Open)
            _socket.Close();

        _socket = null;
    }

    // ── WebSocket callbacks ────────────────────────────────────────────────

    void OnOpen()
    {
        _reconnectAttempt = 0;
        Debug.Log("[EdgePoseReceiver] Connected to edge server.");
    }

    void OnMessage(byte[] data)
    {
        string json = System.Text.Encoding.UTF8.GetString(data);

        InterpretedPoseFrame frame;
        try
        {
            frame = JsonUtility.FromJson<InterpretedPoseFrame>(json);
        }
        catch (Exception ex)
        {
            Debug.LogWarning($"[EdgePoseReceiver] JSON parse error: {ex.Message}");
            return;
        }

        if (frame == null || string.IsNullOrEmpty(frame.user_id)) return;

        // Skip frames that originated from this client (own echo)
        if (string.Equals(frame.user_id, localUserId, StringComparison.Ordinal)) return;

        FramesReceived++;

        if (verboseLogging)
            Debug.Log($"[EdgePoseReceiver] Frame from '{frame.user_id}' " +
                      $"src={frame.prediction_source} α={frame.blend_alpha:F2} " +
                      $"conf={frame.confidence:F2}");

        // Route to the correct remote avatar
        if (_avatarMap.TryGetValue(frame.user_id, out AvatarController controller))
        {
            controller.ApplyPose(frame.ToPoseFrame());
        }
        else if (verboseLogging)
        {
            Debug.LogWarning($"[EdgePoseReceiver] No avatar bound for user_id='{frame.user_id}'. " +
                             "Call RegisterRemoteAvatar() or populate the Inspector list.");
        }
    }

    void OnError(string errorMsg)
    {
        Debug.LogWarning($"[EdgePoseReceiver] WebSocket error: {errorMsg}");
    }

    void OnClose(WebSocketCloseCode closeCode)
    {
        if (_applicationQuitting) return;

        Debug.Log($"[EdgePoseReceiver] Disconnected (code={closeCode}). Scheduling reconnect…");
        _reconnectRoutine = StartCoroutine(ReconnectWithBackoff());
    }

    // ── Reconnection ───────────────────────────────────────────────────────

    IEnumerator ReconnectWithBackoff()
    {
        _reconnectAttempt++;

        // Exponential back-off: delay = base * 2^(attempt-1) + jitter, capped at max
        float delay = Mathf.Min(
            reconnectBaseDelaySec * Mathf.Pow(2f, _reconnectAttempt - 1),
            reconnectMaxDelaySec
        );
        // Add ±20% jitter to avoid thundering herd
        delay *= UnityEngine.Random.Range(0.8f, 1.2f);

        Debug.Log($"[EdgePoseReceiver] Reconnect attempt {_reconnectAttempt} in {delay:F1}s…");
        yield return new WaitForSeconds(delay);

        if (!_applicationQuitting)
            Connect();
    }

    // ── Inner serialisable type for Inspector binding ──────────────────────

    [System.Serializable]
    public class UserAvatarBinding
    {
        [Tooltip("Remote user_id string (must match what the edge server sends).")]
        public string userId;

        [Tooltip("AvatarController to drive with this user's pose.")]
        public AvatarController avatarController;
    }
}
