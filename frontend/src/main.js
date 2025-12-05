document.addEventListener('DOMContentLoaded', () => {
	const form = document.getElementById('predict-form');
	const resultEl = document.getElementById('result');

	function showMessage(msg, isError = false) {
		resultEl.textContent = msg;
		resultEl.className = isError ? 'error' : '';
	}

	form.addEventListener('submit', async (ev) => {
		ev.preventDefault();

		const date = document.getElementById('date').value;
		const volume = Number(document.getElementById('volume').value);
		const open = Number(document.getElementById('open').value);
		const high = Number(document.getElementById('high').value);
		const low = Number(document.getElementById('low').value);

		if (!date) { showMessage('Please pick a date.', true); return; }
		if ([volume, open, high, low].some(v => Number.isNaN(v))) { showMessage('Please provide valid numeric values.', true); return; }

		// split date into year, month, day
		const d = new Date(date + 'T00:00:00');
		const payload = {
			year: d.getUTCFullYear(),
			month: d.getUTCMonth() + 1,
			day: d.getUTCDate(),
			volume,
			open,
			high,
			low
		};

		showMessage('Predicting...');

		try {
			const res = await fetch('/predict', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify(payload)
			});

			if (!res.ok) {
				const text = await res.text().catch(()=>'');
				throw new Error(`Server error: ${res.status} ${text}`);
			}

			const data = await res.json().catch(() => null);
			if (data && (data.predicted_close !== undefined || data.prediction !== undefined)) {
				const pred = data.predicted_close ?? data.prediction;
				showMessage(`Predicted closing price: ${pred}`);
				return;
			}

			// if server returns something unexpected, show it
			if (data) { showMessage(`Server returned: ${JSON.stringify(data)}`); return; }

			throw new Error('Empty response');
		} catch (err) {
			// fallback: simple heuristic if no backend available
			console.warn('Predict failed, using fallback heuristic', err);
			const fallback = ((open + high + low) / 3).toFixed(2);
			showMessage(`(Offline fallback) Predicted closing price (avg of O/H/L): ${fallback}`);
		}
	});
});
