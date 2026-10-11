/* =========================================================
   SurveilHeat AI - Core Client Logic & Interactive Analytics
   ========================================================= */

let map = null;
let heatLayer = null;
let markerLayer = null;
let currentPoints = [];
let domainChart = null;
let cityChart = null;

// City Center Coordinate Lookup
const CITY_COORDS = {
    'Delhi': [28.6139, 77.2090],
    'Mumbai': [19.0760, 72.8777],
    'Bangalore': [12.9716, 77.5946],
    'Hyderabad': [17.3850, 78.4867],
    'Kolkata': [22.5726, 88.3639],
    'Chennai': [13.0827, 80.2707],
    'Pune': [18.5204, 73.8567],
    'Ahmedabad': [23.0225, 72.5714],
    'Jaipur': [26.9124, 75.7873],
    'Lucknow': [26.8467, 80.9462],
    'Kanpur': [26.4499, 80.3319],
    'Surat': [21.1702, 72.8311],
    'Nagpur': [21.1458, 79.0882],
    'Agra': [27.1767, 78.0081],
    'Ludhiana': [30.9010, 75.8573],
    'Visakhapatnam': [17.6868, 83.2185],
    'Thane': [19.2183, 72.9781],
    'Ghaziabad': [28.6692, 77.4538],
    'Indore': [22.7196, 75.8577],
    'Patna': [25.5941, 85.1376],
    'Bhopal': [23.2599, 77.4126],
    'Meerut': [28.9845, 77.7064],
    'Srinagar': [34.0837, 74.7973],
    'Nashik': [19.9975, 73.7898],
    'Vasai': [19.3919, 72.8397],
    'Varanasi': [25.3176, 82.9739],
    'Kalyan': [19.2403, 73.1305],
    'Faridabad': [28.4089, 77.3178],
    'Rajkot': [22.3039, 70.8022]
};

document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initMap();
    fetchStats();
    loadHeatmapData();
    initPredictionForm();
    loadNCRBVerificationCharts();
    loadLeaderboard();
});

/* ---------------- TAB NAVIGATION ---------------- */
function initTabs() {
    const tabBtns = document.querySelectorAll('.tab-btn');
    const panes = document.querySelectorAll('.tab-pane');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => b.classList.remove('active'));
            panes.forEach(p => p.classList.remove('active'));

            btn.classList.add('active');
            const targetId = btn.getAttribute('data-tab');
            const targetPane = document.getElementById(targetId);
            if (targetPane) {
                targetPane.classList.add('active');
                if (targetId === 'heatmap-tab' && map) {
                    map.invalidateSize(true);
                    setTimeout(() => map.invalidateSize(true), 100);
                    setTimeout(() => map.invalidateSize(true), 300);
                    setTimeout(() => map.invalidateSize(true), 600);
                }
            }
        });
    });
}

/* ---------------- MAP INITIALIZATION & HEATMAP ---------------- */
function initMap() {
    const mapElement = document.getElementById('crime-map');
    if (!mapElement) return;

    // Center over India (22.5, 79.0)
    map = L.map('crime-map', {
        center: [22.5, 79.0],
        zoom: 5,
        zoomControl: true,
        preferCanvas: true
    });

    // 1. Cyber Dark Canvas (High-contrast dark mode OpenStreetMap - 100% Free, NO API Key, NO watermark)
    const osmDarkLayer = L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        className: 'cyber-dark-tile',
        maxZoom: 19,
        maxNativeZoom: 19
    });

    // 2. OpenStreetMap Standard (Full street & village details)
    const osmStandardLayer = L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        maxZoom: 19,
        maxNativeZoom: 19
    });

    // 3. OpenStreetMap Humanitarian (High-Contrast Topography)
    const osmHotLayer = L.tileLayer('https://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png', {
        attribution: '&copy; OpenStreetMap contributors, Humanitarian Team',
        maxZoom: 19,
        maxNativeZoom: 18
    });

    // 4. Esri World Street Map (Crisp Vector Roads)
    const esriStreetLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}', {
        attribution: 'Tiles &copy; Esri',
        maxZoom: 19,
        maxNativeZoom: 17
    });

    // 5. Esri Satellite Imagery
    const esriSatelliteLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
        attribution: 'Tiles &copy; Esri',
        maxZoom: 19,
        maxNativeZoom: 16
    });

    // Default to Cyber Dark Canvas
    osmDarkLayer.addTo(map);

    // Layer Switcher Control
    const baseMaps = {
        "🌙 Cyber Dark Canvas": osmDarkLayer,
        "🗺️ Standard India Map": osmStandardLayer,
        "🏔️ High Contrast Topo": osmHotLayer,
        "🏙️ City Street Map": esriStreetLayer,
        "🛰️ Satellite Imagery": esriSatelliteLayer
    };
    L.control.layers(baseMaps).addTo(map);

    markerLayer = L.layerGroup().addTo(map);

    // Heat slider label listener
    const heatSlider = document.getElementById('filter-heat-slider');
    const heatLabel = document.getElementById('heat-val-label');
    if (heatSlider && heatLabel) {
        heatSlider.addEventListener('input', (e) => {
            heatLabel.textContent = e.target.value;
        });
    }

    // Filter submit listener
    const applyBtn = document.getElementById('btn-apply-filters');
    if (applyBtn) {
        applyBtn.addEventListener('click', () => loadHeatmapData());
    }
}

async function loadHeatmapData() {
    const city = document.getElementById('filter-city')?.value || 'all';
    const crimeType = document.getElementById('filter-crime')?.value || 'all';
    const minHeat = document.getElementById('filter-heat-slider')?.value || 0;

    const url = `/api/heatpoints?city=${encodeURIComponent(city)}&crime_type=${encodeURIComponent(crimeType)}&min_heat=${minHeat}&limit=3000`;

    try {
        const res = await fetch(url);
        const data = await res.json();
        
        if (data.status === 'success' && data.points) {
            currentPoints = data.points;
            renderHeatmap(data.points);

            // Update counter in legend
            const countEl = document.getElementById('map-rendered-points');
            if (countEl) {
                countEl.innerHTML = `<i class="fa-solid fa-satellite-dish"></i> ${data.count.toLocaleString()} active surveillance nodes`;
            }

            // Adjust view if specific city is selected
            if (city !== 'all' && CITY_COORDS[city]) {
                map.flyTo(CITY_COORDS[city], 11, { duration: 1.2 });
            } else if (city === 'all') {
                map.flyTo([22.5, 79.0], 5, { duration: 1.0 });
            }
        }
    } catch (err) {
        console.error('Failed to load heatmap data:', err);
    }
}

function renderHeatmap(points) {
    if (!map) return;

    if (heatLayer) {
        map.removeLayer(heatLayer);
    }
    if (markerLayer) {
        markerLayer.clearLayers();
    }

    // Leaflet Heat format: [lat, lon, intensity]
    const heatData = points.map(p => [p.lat, p.lon, p.intensity]);

    heatLayer = L.heatLayer(heatData, {
        radius: 22,
        blur: 18,
        maxZoom: 16,
        max: 1.0,
        gradient: {
            0.15: '#38bdf8',  // Cyan/Blue (Low heat)
            0.45: '#10b981',  // Emerald
            0.65: '#f59e0b',  // Amber
            0.85: '#ef4444',  // Red (High heat / hotspot)
            1.00: '#ff0055'   // Critical Laser Red
        }
    }).addTo(map);

    // Add interactive markers for high-heat hotspots
    const highHeatNodes = points.filter(p => p.heat_level === 'High Heat').slice(0, 50);
    highHeatNodes.forEach(p => {
        const marker = L.circleMarker([p.lat, p.lon], {
            radius: 7,
            fillColor: '#ef4444',
            color: '#ffffff',
            weight: 2,
            opacity: 0.95,
            fillOpacity: 0.85
        });

        marker.bindPopup(`
            <div class="popup-card">
                <div class="popup-header">
                    <span class="popup-badge badge-high">
                        <i class="fa-solid fa-triangle-exclamation"></i> 🚨 HIGH HEAT HOTSPOT
                    </span>
                </div>
                <div class="popup-title">
                    <i class="fa-solid fa-location-dot text-red"></i> ${p.city}
                </div>
                <div style="font-size: 11px; color: #94a3b8; margin-bottom: 6px;">
                    <b>Offense:</b> ${p.crime}
                </div>
                <div class="popup-grid">
                    <div class="popup-stat">
                        <span class="popup-stat-label">Crime Heat Index</span>
                        <span class="popup-stat-val text-red">${p.chi} / 100</span>
                    </div>
                    <div class="popup-stat">
                        <span class="popup-stat-label">Police Deployed</span>
                        <span class="popup-stat-val text-cyan">${p.police} units</span>
                    </div>
                    <div class="popup-stat">
                        <span class="popup-stat-label">Hour Sector</span>
                        <span class="popup-stat-val text-amber">${p.hour}:00 hrs</span>
                    </div>
                    <div class="popup-stat">
                        <span class="popup-stat-label">Status</span>
                        <span class="popup-stat-val text-red">CRITICAL</span>
                    </div>
                </div>
            </div>
        `);
        markerLayer.addLayer(marker);
    });
}

/* ---------------- FETCH SUMMARY STATS ---------------- */
async function fetchStats() {
    try {
        const res = await fetch('/api/stats');
        const data = await res.json();
        if (data.status === 'success') {
            document.getElementById('kpi-total').textContent = data.total_incidents.toLocaleString();
            document.getElementById('kpi-high-heat').textContent = data.high_heat_hotspots.toLocaleString();
            document.getElementById('kpi-avg-chi').textContent = data.avg_crime_heat_index;
            document.getElementById('kpi-closure').textContent = data.case_closure_rate + '%';
        }
    } catch (err) {
        console.error('Error fetching stats:', err);
    }
}

function getHeatMeta(heatLevel) {
    if (heatLevel === 'High Heat') {
        return {
            markerClass: 'marker-high',
            badgeClass: 'badge-high',
            badgeText: '🚨 CRITICAL HIGH HEAT',
            badgeIcon: 'fa-triangle-exclamation',
            color: '#ef4444',
            lightColor: '#ff0055'
        };
    } else if (heatLevel === 'Medium Heat') {
        return {
            markerClass: 'marker-medium',
            badgeClass: 'badge-medium',
            badgeText: '⚠️ ELEVATED MEDIUM HEAT',
            badgeIcon: 'fa-shield-halved',
            color: '#f59e0b',
            lightColor: '#fbbf24'
        };
    } else {
        return {
            markerClass: 'marker-low',
            badgeClass: 'badge-low',
            badgeText: '🟢 LOW RISK / ROUTINE PATROL',
            badgeIcon: 'fa-shield-heart',
            color: '#10b981',
            lightColor: '#34d399'
        };
    }
}

function createRadarTargetMarker(coords, data, locationTitle) {
    const heatLevel = data ? data.predicted_heat_level : 'Low Heat';
    const meta = getHeatMeta(heatLevel);

    const iconHtml = `
        <div class="radar-target-marker ${meta.markerClass}">
            <div class="radar-core"></div>
            <div class="radar-wave wave-1"></div>
            <div class="radar-wave wave-2"></div>
        </div>
    `;

    const customIcon = L.divIcon({
        className: 'radar-div-icon',
        html: iconHtml,
        iconSize: [32, 32],
        iconAnchor: [16, 16],
        popupAnchor: [0, -18]
    });

    const marker = L.marker(coords, { icon: customIcon });

    const violentRisk = data ? data.violent_crime_probability_pct : '25.0';
    const patrols = data ? data.recommended_police_patrols : 2;
    const offense = data && data.input_summary ? data.input_summary['Crime Description'] : 'ROBBERY';
    const weapon = data && data.input_summary ? data.input_summary['Weapon Used'] : 'None';
    const timeOcc = data && data.input_summary ? `${data.input_summary['Hour']}:00 hrs` : '22:00 hrs';

    const popupHtml = `
        <div class="popup-card">
            <div class="popup-header">
                <span class="popup-badge ${meta.badgeClass}">
                    <i class="fa-solid ${meta.badgeIcon}"></i> ${meta.badgeText}
                </span>
            </div>
            <div class="popup-title">
                <i class="fa-solid fa-location-crosshairs" style="color:${meta.color}"></i> ${locationTitle}
            </div>
            <div style="font-size: 11px; color: #94a3b8; margin-bottom: 6px;">
                <b>Incident:</b> ${offense} &bull; <b>Weapon:</b> ${weapon}
            </div>
            <div class="popup-grid">
                <div class="popup-stat">
                    <span class="popup-stat-label">Violent Risk</span>
                    <span class="popup-stat-val" style="color:${meta.color}">${violentRisk}%</span>
                </div>
                <div class="popup-stat">
                    <span class="popup-stat-label">Patrol Units</span>
                    <span class="popup-stat-val text-cyan">${patrols} Units</span>
                </div>
                <div class="popup-stat">
                    <span class="popup-stat-label">Time Sector</span>
                    <span class="popup-stat-val text-amber">${timeOcc}</span>
                </div>
                <div class="popup-stat">
                    <span class="popup-stat-label">Threat Level</span>
                    <span class="popup-stat-val" style="color:${meta.color}">${data ? data.threat_level : 'NORMAL'}</span>
                </div>
            </div>
        </div>
    `;

    marker.bindPopup(popupHtml);
    return marker;
}

let lastPredictedCoords = [28.6139, 77.2090];
let lastPredictedCity = 'Delhi, Delhi (NCT)';
let lastPredictedData = null;
let currentTargetMarker = null;

/* ---------------- AI RISK PREDICTION SIMULATOR ---------------- */
function initPredictionForm() {
    const form = document.getElementById('prediction-form');
    const policeRange = document.getElementById('pred-police');
    const policeVal = document.getElementById('pred-police-val');
    const stateInput = document.getElementById('pred-state');
    const cityInput = document.getElementById('pred-city');
    const viewMapBtn = document.getElementById('btn-view-pred-map');

    if (policeRange && policeVal) {
        policeRange.addEventListener('input', (e) => {
            policeVal.textContent = e.target.value;
        });
    }

    if (viewMapBtn) {
        viewMapBtn.addEventListener('click', () => {
            // 1. Switch to Heatmap Tab
            const heatmapTabBtn = document.getElementById('btn-tab-heatmap');
            if (heatmapTabBtn) heatmapTabBtn.click();

            // 2. Trigger multi-stage invalidateSize to ensure tiles render immediately
            [50, 150, 300, 500].forEach(delay => {
                setTimeout(() => {
                    if (map) map.invalidateSize(true);
                }, delay);
            });

            // 3. Smooth camera flight to targeted coordinates
            setTimeout(() => {
                if (map && lastPredictedCoords) {
                    map.invalidateSize(true);
                    map.flyTo(lastPredictedCoords, 12, { duration: 1.2 });

                    if (markerLayer) {
                        if (currentTargetMarker) {
                            markerLayer.removeLayer(currentTargetMarker);
                        }

                        // Create dynamically classified radar target marker
                        currentTargetMarker = createRadarTargetMarker(
                            lastPredictedCoords,
                            lastPredictedData,
                            lastPredictedCity
                        );

                        markerLayer.addLayer(currentTargetMarker);
                        setTimeout(() => currentTargetMarker.openPopup(), 800);
                    }
                }
            }, 180);
        });
    }

    if (form) {
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            const btn = document.getElementById('btn-run-predict');
            btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Analyzing Surveillance Heat...';
            btn.disabled = true;

            const selectedState = stateInput.value.trim() || 'Delhi (NCT)';
            const selectedCity = cityInput.value.trim() || 'Delhi';

            const payload = {
                state: selectedState,
                city: selectedCity,
                crime_desc: document.getElementById('pred-crime').value,
                weapon: document.getElementById('pred-weapon').value,
                hour: parseInt(document.getElementById('pred-hour').value),
                day_of_week: parseInt(document.getElementById('pred-dow').value),
                month: parseInt(document.getElementById('pred-month').value),
                victim_age: parseInt(document.getElementById('pred-age').value),
                victim_gender: document.getElementById('pred-gender').value,
                police_deployed: parseInt(document.getElementById('pred-police').value)
            };

            try {
                const res = await fetch('/api/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const responseData = await res.json();

                if (responseData.status === 'success') {
                    lastPredictedData = responseData.data;
                    if (responseData.data.city_coordinates) {
                        lastPredictedCoords = [
                            responseData.data.city_coordinates.lat,
                            responseData.data.city_coordinates.lon
                        ];
                        lastPredictedCity = `${selectedCity}, ${selectedState}`;
                    }
                    renderPredictionResult(responseData.data);
                }
            } catch (err) {
                console.error('Prediction error:', err);
            } finally {
                btn.innerHTML = '<i class="fa-solid fa-microchip"></i> Execute ML Risk & Heat Inference';
                btn.disabled = false;
            }
        });
    }
}

function renderPredictionResult(data) {
    const statusEl = document.getElementById('res-threat-status');
    const heatEl = document.getElementById('res-heat-level');
    const jurisdictionEl = document.getElementById('res-jurisdiction');
    const probHighVal = document.getElementById('prob-high-val');
    const probMedVal = document.getElementById('prob-med-val');
    const probLowVal = document.getElementById('prob-low-val');
    const probHighBar = document.getElementById('prob-high-bar');
    const probMedBar = document.getElementById('prob-med-bar');
    const probLowBar = document.getElementById('prob-low-bar');
    const violentEl = document.getElementById('res-violent-risk');
    const policeEl = document.getElementById('res-rec-police');
    const ncrbEl = document.getElementById('res-ncrb-share');

    if (jurisdictionEl && data.city) {
        jurisdictionEl.innerHTML = `<i class="fa-solid fa-location-dot"></i> ${data.city}, ${data.state || ''}`;
    }

    heatEl.textContent = data.predicted_heat_level;
    statusEl.innerHTML = `<i class="fa-solid fa-triangle-exclamation"></i> ${data.threat_level}`;

    if (data.predicted_heat_level === 'High Heat') {
        heatEl.className = 'text-red';
        statusEl.className = 'threat-alert red-alert';
    } else if (data.predicted_heat_level === 'Medium Heat') {
        heatEl.className = 'text-amber';
        statusEl.className = 'threat-alert badge-amber';
    } else {
        heatEl.className = 'text-cyan';
        statusEl.className = 'threat-alert badge-green';
    }

    const pHigh = data.heat_probabilities['High Heat'] || 0;
    const pMed = data.heat_probabilities['Medium Heat'] || 0;
    const pLow = data.heat_probabilities['Low Heat'] || 0;

    probHighVal.textContent = pHigh + '%';
    probMedVal.textContent = pMed + '%';
    probLowVal.textContent = pLow + '%';

    probHighBar.style.width = pHigh + '%';
    probMedBar.style.width = pMed + '%';
    probLowBar.style.width = pLow + '%';

    violentEl.textContent = data.violent_crime_probability_pct + '%';
    policeEl.textContent = data.recommended_police_patrols + ' Units';
    ncrbEl.textContent = data.ncrb_city_share_pct + '%';
}

/* ---------------- NCRB VERIFICATION CHARTS ---------------- */
async function loadNCRBVerificationCharts() {
    try {
        const res = await fetch('/api/ncrb_benchmark');
        const json = await res.json();
        if (json.status !== 'success' || !json.data) return;

        const ncrb = json.data;

        // 1. Crime Domain Distribution Doughnut Chart
        const domainCtx = document.getElementById('chart-domain-dist')?.getContext('2d');
        if (domainCtx) {
            const domainLabels = Object.keys(ncrb.domain_breakdown);
            const predDomainVals = domainLabels.map(k => ncrb.domain_breakdown[k].predicted_share_pct);
            const ncrbDomainVals = domainLabels.map(k => ncrb.domain_breakdown[k].ncrb_benchmark_pct);

            if (domainChart) domainChart.destroy();
            domainChart = new Chart(domainCtx, {
                type: 'doughnut',
                data: {
                    labels: domainLabels,
                    datasets: [{
                        label: 'Observed / Model Share (%)',
                        data: predDomainVals,
                        backgroundColor: ['#38bdf8', '#ef4444', '#f59e0b', '#a855f7'],
                        borderWidth: 2,
                        borderColor: '#0f172a'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'bottom',
                            labels: { color: '#94a3b8', font: { family: 'Outfit', size: 11 } }
                        }
                    }
                }
            });
        }

        // 2. City Share vs NCRB Expected Grouped Bar Chart (Top 8 Cities)
        const cityCtx = document.getElementById('chart-city-ncrb')?.getContext('2d');
        if (cityCtx) {
            const topCities = ['Delhi', 'Mumbai', 'Bangalore', 'Hyderabad', 'Kolkata', 'Chennai', 'Pune', 'Ahmedabad'];
            const predCityVals = topCities.map(c => ncrb.city_breakdown[c]?.predicted_share_pct || 0);
            const ncrbCityVals = topCities.map(c => ncrb.city_breakdown[c]?.ncrb_benchmark_pct || 0);

            if (cityChart) cityChart.destroy();
            cityChart = new Chart(cityCtx, {
                type: 'bar',
                data: {
                    labels: topCities,
                    datasets: [
                        {
                            label: 'Predicted / Model Share (%)',
                            data: predCityVals,
                            backgroundColor: '#00f2fe',
                            borderRadius: 6
                        },
                        {
                            label: 'NCRB Official Benchmark (%)',
                            data: ncrbCityVals,
                            backgroundColor: '#f59e0b',
                            borderRadius: 6
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                        y: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } }
                    },
                    plugins: {
                        legend: {
                            labels: { color: '#94a3b8', font: { family: 'Outfit' } }
                        }
                    }
                }
            });
        }
    } catch (err) {
        console.error('Error loading NCRB charts:', err);
    }
}

/* ---------------- CITY HOTSPOT LEADERBOARD ---------------- */
async function loadLeaderboard() {
    try {
        const res = await fetch('/api/city_summary');
        const json = await res.json();
        if (json.status !== 'success' || !json.cities) return;

        const tbody = document.getElementById('leaderboard-tbody');
        if (!tbody) return;

        const renderRows = (cities) => {
            tbody.innerHTML = '';
            cities.forEach((c, idx) => {
                const tr = document.createElement('tr');
                const statusBadge = c.AvgCHI >= 23 
                    ? '<span class="status-badge badge-red">High Surveillance</span>'
                    : (c.AvgCHI >= 19 ? '<span class="status-badge badge-amber">Active Patrol</span>' : '<span class="status-badge badge-green">Standard</span>');

                tr.innerHTML = `
                    <td><strong>#${idx + 1}</strong></td>
                    <td><strong>${c.City}</strong></td>
                    <td>${c.IncidentCount.toLocaleString()}</td>
                    <td><strong class="${c.AvgCHI >= 22 ? 'text-red' : 'text-amber'}">${c.AvgCHI}</strong></td>
                    <td>${c.ViolentCrimeRate}%</td>
                    <td>${c.AvgPolice} units</td>
                    <td>${statusBadge}</td>
                    <td>
                        <button class="btn-locate" onclick="focusCityOnMap('${c.City}')">
                            <i class="fa-solid fa-crosshairs"></i> Focus Map
                        </button>
                    </td>
                `;
                tbody.appendChild(tr);
            });
        };

        renderRows(json.cities);

        // Search filter listener
        const searchInput = document.getElementById('leaderboard-search');
        if (searchInput) {
            searchInput.addEventListener('input', (e) => {
                const query = e.target.value.toLowerCase();
                const filtered = json.cities.filter(c => c.City.toLowerCase().includes(query));
                renderRows(filtered);
            });
        }
    } catch (err) {
        console.error('Error loading leaderboard:', err);
    }
}

async function focusCityOnMap(cityName) {
    if (!map) return;
    
    // Switch to Heatmap Tab
    const heatmapTabBtn = document.getElementById('btn-tab-heatmap');
    if (heatmapTabBtn) heatmapTabBtn.click();

    // Select city in filter dropdown if present
    const citySelect = document.getElementById('filter-city');
    if (citySelect) {
        citySelect.value = cityName;
    }

    try {
        const res = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ city: cityName })
        });
        const resp = await res.json();
        if (resp.status === 'success' && resp.data.city_coordinates) {
            const coords = [resp.data.city_coordinates.lat, resp.data.city_coordinates.lon];
            setTimeout(() => {
                map.invalidateSize();
                map.flyTo(coords, 13, { duration: 1.4 });

                if (markerLayer) {
                    if (currentTargetMarker) {
                        markerLayer.removeLayer(currentTargetMarker);
                    }
                    currentTargetMarker = createRadarTargetMarker(
                        coords, 
                        resp.data, 
                        `${cityName}, ${resp.data.state || ''}`
                    );
                    markerLayer.addLayer(currentTargetMarker);
                    setTimeout(() => currentTargetMarker.openPopup(), 1100);
                }
            }, 180);
        }
    } catch (e) {
        console.error('Focus city error:', e);
    }
}
