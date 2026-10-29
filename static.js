
// ==========================================
// VisionGuard - Dashboard JavaScript
// ==========================================

// Update a text element safely
function updateText(elementId, value, fallback = "Unknown") {
    const element = document.getElementById(elementId);

    if (element) {
        element.textContent =
            value !== undefined && value !== null && value !== ""
                ? value
                : fallback;
    }
}

// Update a status indicator's color
function updateIndicator(elementId, state) {
    const element = document.getElementById(elementId);

    if (!element) return;

    element.classList.remove(
        "indicator-normal",
        "indicator-warning",
        "indicator-danger",
        "indicator-neutral"
    );

    const value = String(state || "").toLowerCase();

    if (
        value.includes("warning") ||
        value.includes("drowsy") ||
        value.includes("detected") ||
        value.includes("down") ||
        value.includes("tilted") ||
        value.includes("closed")
    ) {
        element.classList.add("indicator-warning");
    } else if (
        value.includes("danger") ||
        value.includes("critical")
    ) {
        element.classList.add("indicator-danger");
    } else if (
        value.includes("open") ||
        value.includes("normal") ||
        value.includes("alert") ||
        value.includes("no yawn")
    ) {
        element.classList.add("indicator-normal");
    } else {
        element.classList.add("indicator-neutral");
    }
}

// Update the overall monitoring status
function updateOverallStatus(status) {
    const statusElement = document.getElementById("overall-status");
    const statusCard = document.getElementById("overall-status-card");

    const value = String(status || "Waiting...").toLowerCase();

    if (statusElement) {
        statusElement.textContent = status || "Waiting...";
    }

    if (statusCard) {
        statusCard.classList.remove(
            "status-normal",
            "status-warning",
            "status-danger"
        );

        if (
            value.includes("drowsy") ||
            value.includes("danger") ||
            value.includes("critical")
        ) {
            statusCard.classList.add("status-danger");
        } else if (
            value.includes("warning") ||
            value.includes("fatigue") ||
            value.includes("check")
        ) {
            statusCard.classList.add("status-warning");
        } else {
            statusCard.classList.add("status-normal");
        }
    }
}

// Fetch detection data from Flask
async function updateDashboard() {
    try {
        const response = await fetch("/status", {
            method: "GET",
            cache: "no-store"
        });

        if (!response.ok) {
            throw new Error(
                `Server returned status ${response.status}`
            );
        }

        const data = await response.json();

        // Eye status
        updateText("eye-status", data.eye_status);
        updateIndicator("eye-indicator", data.eye_status);

        // Yawn status
        updateText("yawn-status", data.yawn_status);
        updateIndicator("yawn-indicator", data.yawn_status);

        // Head position
        updateText("head-status", data.head_status);
        updateIndicator("head-indicator", data.head_status);

        // Eye Aspect Ratio
        const ear = Number(data.ear);
        updateText(
            "ear-value",
            Number.isFinite(ear) ? ear.toFixed(3) : "0.000"
        );

        // Mouth Aspect Ratio
        const mar = Number(data.mar);
        updateText(
            "mar-value",
            Number.isFinite(mar) ? mar.toFixed(3) : "0.000"
        );

        // Overall monitoring status
        updateOverallStatus(data.overall_status);

        // Face detection status, if this element exists
        if (typeof data.face_detected !== "undefined") {
            updateText(
                "face-status",
                data.face_detected ? "Face detected" : "No face detected"
            );
        }

    } catch (error) {
        console.error("VisionGuard dashboard error:", error);

        updateText("overall-status", "Connection Error");
    }
}

// Start dashboard updates
updateDashboard();

// Refresh every 500 milliseconds
const dashboardInterval = setInterval(updateDashboard, 500);

// Stop polling when the page is closed
window.addEventListener("beforeunload", function () {
    clearInterval(dashboardInterval);
});
