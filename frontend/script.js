document.addEventListener("DOMContentLoaded", () => {
	// --- DOM Elements ---
	const homeTeamSelect = document.getElementById("home-team");
	const awayTeamSelect = document.getElementById("away-team");
	const predictButton = document.getElementById("predict-button");
	const predictionResultDiv = document.getElementById("prediction-result");

	// --- Configuration ---
	const API_BASE_URL = "http://localhost:5000/api";

	// --- Data ---
	let teams = []; // Will be loaded from API

	// --- Functions ---

	/**
	 * Loads teams from the backend API.
	 */
	async function loadTeams() {
		try {
			console.log("🔄 Loading teams from API...");
			const response = await fetch(`${API_BASE_URL}/teams`);

			if (!response.ok) {
				throw new Error(`HTTP ${response.status}: ${response.statusText}`);
			}

			const data = await response.json();

			if (data.success) {
				teams = data.teams;
				console.log(`✅ Loaded ${teams.length} teams from database`);
				populateDropdowns();
			} else {
				throw new Error(data.error || "Failed to load teams");
			}
		} catch (error) {
			console.error("❌ Error loading teams:", error);
			displayError(`Failed to load teams: ${error.message}`);

			// Fallback to hardcoded teams if API fails
			teams = [
				"Arsenal",
				"Aston Villa",
				"Bournemouth",
				"Brentford",
				"Brighton & Hove Albion",
				"Burnley",
				"Chelsea",
				"Crystal Palace",
				"Everton",
				"Fulham",
				"Liverpool",
				"Luton Town",
				"Manchester City",
				"Manchester United",
				"Newcastle United",
				"Nottingham Forest",
				"Sheffield United",
				"Tottenham Hotspur",
				"West Ham United",
				"Wolverhampton Wanderers",
			];
			console.log("🔄 Using fallback team list");
			populateDropdowns();
		}
	}

	/**
	 * Populates both dropdown select elements with the loaded teams.
	 */
	function populateDropdowns() {
		// Clear existing options
		homeTeamSelect.innerHTML = "";
		awayTeamSelect.innerHTML = "";

		// Add default option
		const defaultOption = document.createElement("option");
		defaultOption.value = "";
		defaultOption.textContent = "Select a team...";
		homeTeamSelect.appendChild(defaultOption.cloneNode(true));
		awayTeamSelect.appendChild(defaultOption.cloneNode(true));

		// Add team options
		teams.sort().forEach((team) => {
			const homeOption = document.createElement("option");
			homeOption.value = team;
			homeOption.textContent = team;
			homeTeamSelect.appendChild(homeOption);

			const awayOption = document.createElement("option");
			awayOption.value = team;
			awayOption.textContent = team;
			awayTeamSelect.appendChild(awayOption);
		});
	}

	/**
	 * REAL AI PREDICTION using backend API
	 * Calls the actual machine learning models to predict match outcomes.
	 */
	async function predictOutcome(homeTeam, awayTeam, modelType = "neural") {
		try {
			console.log(
				`🤖 Making AI prediction: ${homeTeam} vs ${awayTeam} (${modelType})`
			);

			const response = await fetch(`${API_BASE_URL}/predict`, {
				method: "POST",
				headers: {
					"Content-Type": "application/json",
				},
				body: JSON.stringify({
					homeTeam: homeTeam,
					awayTeam: awayTeam,
					modelType: modelType,
				}),
			});

			if (!response.ok) {
				throw new Error(`HTTP ${response.status}: ${response.statusText}`);
			}

			const data = await response.json();

			if (data.success) {
				console.log("✅ Prediction successful:", data.prediction);
				return {
					homeWin: data.prediction.homeWin,
					draw: data.prediction.draw,
					awayWin: data.prediction.awayWin,
					modelInfo: data.modelInfo,
					confidence: data.modelInfo.confidence,
				};
			} else {
				throw new Error(data.error || "Prediction failed");
			}
		} catch (error) {
			console.error("❌ Prediction error:", error);
			throw error;
		}
	}

	/**
	 * Displays the AI prediction probabilities in the UI.
	 * @param {string} homeTeam
	 * @param {string} awayTeam
	 * @param {object} result - The prediction result object with AI model info.
	 */
	function displayResult(homeTeam, awayTeam, result) {
		const modelBadge = result.modelInfo
			? `<span class="model-badge ${result.modelInfo.type}">${result.modelInfo.name}</span>`
			: "";

		const confidenceBadge = result.confidence
			? `<span class="confidence-badge ${result.confidence.toLowerCase()}">${
					result.confidence
			  } Confidence</span>`
			: "";

		predictionResultDiv.innerHTML = `
      <div class="prediction-header">
        <h3>🤖 AI Prediction Results</h3>
        <div class="model-info">
          ${modelBadge}
          ${confidenceBadge}
        </div>
      </div>
      <div class="probability-container">
        <div class="prob-item ${
									result.homeWin >= Math.max(result.draw, result.awayWin) ? "winner" : ""
								}">
          <div class="percentage">${result.homeWin}%</div>
          <div class="label">${homeTeam} Win</div>
        </div>
        <div class="prob-item ${
									result.draw >= Math.max(result.homeWin, result.awayWin) ? "winner" : ""
								}">
          <div class="percentage">${result.draw}%</div>
          <div class="label">Draw</div>
        </div>
        <div class="prob-item ${
									result.awayWin >= Math.max(result.homeWin, result.draw) ? "winner" : ""
								}">
          <div class="percentage">${result.awayWin}%</div>
          <div class="label">${awayTeam} Win</div>
        </div>
      </div>
      <div class="prediction-note">
        <small>⚡ Powered by ${
									result.modelInfo?.type === "neural"
										? "Deep Learning Neural Network (195K parameters)"
										: "Traditional Machine Learning"
								}</small>
      </div>
    `;
		predictionResultDiv.classList.remove("hidden");
		// Use a tiny timeout to ensure the CSS transition triggers
		setTimeout(() => {
			predictionResultDiv.classList.add("visible");
		}, 10);
	}

	/**
	 * Displays an error message in the result area.
	 * @param {string} message - The error message to display.
	 */
	function displayError(message) {
		predictionResultDiv.innerHTML = `
      <h3>Error</h3>
      <p style="font-size: 1.1rem; color: #ff8a8a;">${message}</p>
    `;
		predictionResultDiv.classList.remove("hidden");
		setTimeout(() => {
			predictionResultDiv.classList.add("visible");
		}, 10);
	}

	/**
	 * Handles the click event for the predict button.
	 */
	async function handlePredictionClick() {
		const homeTeam = homeTeamSelect.value;
		const awayTeam = awayTeamSelect.value;

		// Validation
		if (!homeTeam || !awayTeam) {
			displayError("Please select both teams.");
			return;
		}

		if (homeTeam === awayTeam) {
			displayError("Please select two different teams.");
			return;
		}

		// Show loading state on button
		predictButton.disabled = true;
		predictButton.innerHTML = `
      <svg class="spinner" viewBox="0 0 50 50"><circle class="path" cx="25" cy="25" r="20" fill="none" stroke-width="5"></circle></svg>
      <span>🤖 AI Analyzing...</span>`;

		// Hide previous results
		predictionResultDiv.classList.remove("visible");

		try {
			// Get AI prediction (try neural network first, fallback to traditional)
			let result;
			try {
				result = await predictOutcome(homeTeam, awayTeam, "neural");
			} catch (neuralError) {
				console.log("Neural network failed, trying traditional ML...");
				result = await predictOutcome(homeTeam, awayTeam, "traditional");
			}

			// Display new result
			displayResult(homeTeam, awayTeam, result);
		} catch (error) {
			console.error("❌ Both models failed:", error);
			displayError(`AI prediction failed: ${error.message}`);
		}

		// Restore button state
		predictButton.disabled = false;
		predictButton.innerHTML = `
      <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>
      <span>Predict Outcome</span>`;
	}

	// --- Initialization ---
	async function initializeApp() {
		// Load teams from API
		await loadTeams();

		// Set default different teams if available
		if (teams.length > 0) {
			const arsenalIndex = teams.findIndex((team) => team.includes("Arsenal"));
			const cityIndex = teams.findIndex((team) =>
				team.includes("Manchester City")
			);

			if (arsenalIndex !== -1) homeTeamSelect.selectedIndex = arsenalIndex + 1; // +1 for default option
			if (cityIndex !== -1) awayTeamSelect.selectedIndex = cityIndex + 1;
		}
	}

	// Start the app
	initializeApp();

	// Add a loading spinner style for the button
	const style = document.createElement("style");
	style.innerHTML = `
    .spinner { animation: rotate 2s linear infinite; width: 24px; height: 24px; }
    .spinner .path { stroke: #fff; stroke-linecap: round; animation: dash 1.5s ease-in-out infinite; }
    @keyframes rotate { 100% { transform: rotate(360deg); } }
    @keyframes dash { 0% { stroke-dasharray: 1, 150; stroke-dashoffset: 0; } 50% { stroke-dasharray: 90, 150; stroke-dashoffset: -35; } 100% { stroke-dasharray: 90, 150; stroke-dashoffset: -124; } }
  `;
	document.head.appendChild(style);

	// --- Event Listeners ---
	predictButton.addEventListener("click", handlePredictionClick);
});
