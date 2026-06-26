/*
==========================================================
BISINDO Detection System
app.js
==========================================================
*/

const video = document.getElementById("camera");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const gestureText = document.getElementById("gesture");
const confidenceText = document.getElementById("confidence");
const fpsText = document.getElementById("fps");
const statusText = document.getElementById("status");

const startButton = document.getElementById("start-btn");
const stopButton = document.getElementById("stop-btn");

const modeRadio = document.querySelectorAll(
    'input[name="mode"]'
);

let socket = null;
let stream = null;
let timer = null;

let lastTime = performance.now();

// ==========================================================
// WebSocket
// ==========================================================

function connectWebSocket() {

    socket = new WebSocket(
        `ws://${location.host}/ws`
    );

    socket.binaryType = "arraybuffer";

    socket.onopen = () => {

        statusText.textContent = "Connected";

        sendMode();

    };

    socket.onclose = () => {

        statusText.textContent = "Disconnected";

    };

    socket.onerror = () => {

        statusText.textContent = "Error";

    };

    socket.onmessage = (event) => {

        const result = JSON.parse(
            event.data
        );

        gestureText.textContent =
            result.label;

        confidenceText.textContent =
            `${(result.confidence * 100).toFixed(2)} %`;

        const now = performance.now();

        fpsText.textContent =
            (1000 / (now - lastTime)).toFixed(1);

        lastTime = now;

    };

}

// ==========================================================
// Camera
// ==========================================================

async function startCamera() {

    stream =
        await navigator.mediaDevices.getUserMedia({

            video: true,

            audio: false

        });

    video.srcObject = stream;

}

function stopCamera() {

    if (timer) {

        clearInterval(timer);

    }

    if (stream) {

        stream.getTracks().forEach(track => {

            track.stop();

        });

    }

    if (socket) {

        socket.close();

    }

}

// ==========================================================
// Send Mode
// ==========================================================

function currentMode() {

    return document.querySelector(
        'input[name="mode"]:checked'
    ).value;

}

function sendMode() {

    if (

        socket &&

        socket.readyState === WebSocket.OPEN

    ) {

        socket.send(

            currentMode()

        );

    }

}

// ==========================================================
// Send Frame
// ==========================================================

function sendFrame() {

    if (

        !socket ||

        socket.readyState !== WebSocket.OPEN

    ) {

        return;

    }

    canvas.width = video.videoWidth;

    canvas.height = video.videoHeight;

    ctx.drawImage(

        video,

        0,

        0,

        canvas.width,

        canvas.height

    );

    canvas.toBlob(

        blob => {

            if (!blob) return;

            blob.arrayBuffer()

                .then(buffer => {

                    socket.send(buffer);

                });

        },

        "image/jpeg",

        0.8

    );

}

// ==========================================================
// Events
// ==========================================================

startButton.addEventListener(

    "click",

    async () => {

        await startCamera();

        connectWebSocket();

        timer = setInterval(

            sendFrame,

            100

        );

    }

);

stopButton.addEventListener(

    "click",

    stopCamera

);

modeRadio.forEach(

    radio => {

        radio.addEventListener(

            "change",

            sendMode

        );

    }

);
