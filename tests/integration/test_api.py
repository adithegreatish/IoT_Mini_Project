"""Integration tests for FastAPI endpoints."""

import pytest
from httpx import ASGITransport, AsyncClient
from app.main import app, lifespan
from app.config.loader import load_config


@pytest.mark.asyncio
async def test_api_endpoints():
    load_config("configs/windows.yaml")
    async with lifespan(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # 1. Test root dashboard HTML
            resp = await client.get("/")
            assert resp.status_code == 200

            # 2. Test /api/health
            resp = await client.get("/api/health")
            assert resp.status_code == 200
            data = resp.json()
            assert "uptime_sec" in data
            assert "platform" in data

            # 3. Test /api/camera/status
            resp = await client.get("/api/camera/status")
            assert resp.status_code == 200
            cam = resp.json()
            assert "negotiated_resolution" in cam
            assert "observed_fps" in cam

            # 4. Test preview pause / resume
            resp = await client.post("/api/camera/pause_preview")
            assert resp.status_code == 200
            assert resp.json()["preview_paused"] is True

            resp = await client.post("/api/camera/resume_preview")
            assert resp.status_code == 200
            assert resp.json()["preview_paused"] is False

            # 5. Test slots read and update
            resp = await client.get("/api/slots")
            assert resp.status_code == 200
            slots_data = resp.json()
            assert "slots" in slots_data

            update_payload = {
                "reference_width": 1280,
                "reference_height": 720,
                "slots": [
                    {
                        "id": "T01",
                        "label": "Test Bay",
                        "slot_type": "car",
                        "enabled": True,
                        "points": [[0.1, 0.1], [0.3, 0.1], [0.3, 0.4], [0.1, 0.4]],
                    }
                ],
            }
            resp = await client.post("/api/slots", json=update_payload)
            assert resp.status_code == 200
            assert resp.json()["slots_count"] == 1

            # 6. Test models list
            resp = await client.get("/api/models")
            assert resp.status_code == 200
            models_data = resp.json()
            assert "models" in models_data
            assert len(models_data["models"]) >= 1

            # 7. Test config read
            resp = await client.get("/api/config")
            assert resp.status_code == 200
            assert resp.json()["app"]["env"] == "windows"

            # 8. Test parking availability and virtual allocation
            resp = await client.get("/api/parking/availability")
            assert resp.status_code == 200
            avail_data = resp.json()
            assert "total_slots" in avail_data
            assert "physical_counts" in avail_data
            assert "reservation_counts" in avail_data
            assert "by_slot_type" in avail_data
            assert "by_zone" in avail_data

            # 9. Test parking sync slots
            resp = await client.post("/api/parking/sync_slots")
            assert resp.status_code == 200
            assert "synchronized_slots" in resp.json()

            # 10. Test parking audit trail
            resp = await client.get("/api/parking/audit")
            assert resp.status_code == 200
            assert "audit_trail" in resp.json()

