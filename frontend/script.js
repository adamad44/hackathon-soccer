document.addEventListener("DOMContentLoaded", () => {
	// --- DOM Elements ---
	const homeTeamSelect = document.getElementById("home-team");
	const awayTeamSelect = document.getElementById("away-team");
	const predictButton = document.getElementById("predict-button");
	const predictionResultDiv = document.getElementById("prediction-result");

	// --- Data ---
	const teams = [
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

	// --- Functions ---

	/**
	 * Populates a dropdown select element with a list of teams.
	 * @param {HTMLSelectElement} selectElement - The dropdown to populate.
	 */
	function populateDropdown(selectElement) {
		teams.sort().forEach((team) => {
			const option = document.createElement("option");
			option.value = team;
			option.textContent = team;
			selectElement.appendChild(option);
		});
	}

	/**
	 * PLACEHOLDER PREDICTION LOGIC
	 * Generates random percentages for win, draw, and loss.
	 * --- REPLACE THIS with your actual machine learning model call. ---
	 */
	function predictOutcome(homeTeam, awayTeam) {
		// Simulate a network delay for a more "realistic" feel
		return new Promise((resolve) => {
			setTimeout(() => {
				// Generate two random numbers to slice the 100% into three parts
				const break1 = Math.random() * 100;
				const break2 = Math.random() * 100;

				const points = [0, break1, break2, 100].sort((a, b) => a - b);

				const homeWin = Math.round(points[1] - points[0]);
				const draw = Math.round(points[2] - points[1]);
				const awayWin = Math.round(points[3] - points[2]);

				// Resolve the promise with the probability object
				resolve({ homeWin, draw, awayWin });
			}, 750); // 0.75 second delay
		});
	}

	/**
	 * Displays the prediction probabilities in the UI.
	 * @param {string} homeTeam
	 * @param {string} awayTeam
	 * @param {object} result - The prediction result object { homeWin, draw, awayWin }.
	 */
	function displayResult(homeTeam, awayTeam, result) {
		predictionResultDiv.innerHTML = `
      <h3>Prediction Probabilities</h3>
      <div class="probability-container">
        <div class="prob-item">
          <div class="percentage">${result.homeWin}%</div>
          <div class="label">${homeTeam} Win</div>
        </div>
        <div class="prob-item">
          <div class="percentage">${result.draw}%</div>
          <div class="label">Draw</div>
        </div>
        <div class="prob-item">
          <div class="percentage">${result.awayWin}%</div>
          <div class="label">${awayTeam} Win</div>
        </div>
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
		if (homeTeam === awayTeam) {
			displayError("Please select two different teams.");
			return;
		}

		// Show loading state on button
		predictButton.disabled = true;
		predictButton.innerHTML = `
      <svg class="spinner" viewBox="0 0 50 50"><circle class="path" cx="25" cy="25" r="20" fill="none" stroke-width="5"></circle></svg>
      <span>Analyzing...</span>`;

		// Hide previous results
		predictionResultDiv.classList.remove("visible");

		// Get prediction probabilities
		const result = await predictOutcome(homeTeam, awayTeam);

		// Display new result
		displayResult(homeTeam, awayTeam, result);

		// Restore button state
		predictButton.disabled = false;
		predictButton.innerHTML = `
      <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>
      <span>Predict Outcome</span>`;
	}

	// --- Initialization ---
	populateDropdown(homeTeamSelect);
	populateDropdown(awayTeamSelect);

	// Set default different teams
	homeTeamSelect.selectedIndex = 0; // Arsenal
	awayTeamSelect.selectedIndex = 12; // Manchester City

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
