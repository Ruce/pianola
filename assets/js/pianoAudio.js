class PianoAudio {
	constructor(bpm, pianoKeys) {
		this.bpm = bpm;
		this.sampler = this.initialiseSampler(pianoKeys);
		this.toneStarted = false;
	}
	
	static circVar(samples, high=1, low=0) {
		let sinTotal = 0;
		let cosTotal = 0;
		for (const n of samples) {
			sinTotal += Math.sin(((n - low) * 2 * Math.PI) / (high - low));
			cosTotal += Math.cos(((n - low) * 2 * Math.PI) / (high - low));
		}
		const sinMean = sinTotal / samples.length;
		const cosMean = cosTotal / samples.length;
		const hypot = Math.sqrt(Math.pow(sinMean, 2) + Math.pow(cosMean, 2));
		
		return Math.min(1 - hypot, 1);
	}

	static calcDispersion(noteTimings, interval) {
		const deltas = Array.from(noteTimings, (t) => (t % interval) / interval);
		return PianoAudio.circVar(deltas, 1, 0);
	}

	static calcBestInterval(noteTimings, low, high, bias=0) {
		// `bias`: add a bias to the dispersion of lower intervals to prefer high intervals
		let lowestDispersion = null;
		let bestInterval = -1;
		
		for (let i = low; i <= high; i++) {
			let dispersion = PianoAudio.calcDispersion(noteTimings, i) + (bias * (high - i));
			if (lowestDispersion === null || dispersion < lowestDispersion) {
				lowestDispersion = dispersion;
				bestInterval = i;
			}
		}
		return bestInterval;
	}
	
	static detectBpm(noteHistory, lowestBpm, highestBpm) {
		const ticksPerSec = 480;
		const noteTimings = Array.from(noteHistory, (n) => n.time * ticksPerSec);
		const lowInterval = Math.floor(ticksPerSec * (60 / highestBpm) / 4);
		const highInterval = Math.ceil(ticksPerSec * (60 / lowestBpm) / 4);
		
		const bestInterval = PianoAudio.calcBestInterval(noteTimings, lowInterval, highInterval, 0.002);
		const bestBpm = Math.round(ticksPerSec * 60 / (bestInterval * 4));
		console.log('Detected bpm:', bestBpm);
		return bestBpm;
	}

	setBPM(bpm) {
		Tone.Transport.bpm.value = bpm;
		this.bpm = bpm;
	}

	initialiseSampler(pianoKeys) {
		const noteKeys = pianoKeys.map((k) => k.keyName); // Get a list of all notes e.g. ['A3', 'A#3', 'B3', 'C4'...]
		const sampleFiles = Object.assign({}, ...noteKeys.map((n) => ({[n]: n.replace('#', 's') + ".mp3"})));
		// No sample files for keys A0, A#0, and B0
		delete sampleFiles['A0'];
		delete sampleFiles['A#0'];
		delete sampleFiles['B0'];
		
		const sampler = new Tone.Sampler({
			urls: sampleFiles,
			baseUrl: "assets/samples/piano/",
			release: 0.3,
			volume: -8
		}).toDestination();
		
		return sampler;
	}
  
	changeVolume(volume) {
		const volumeDb = (volume < 1) ? -Infinity : -(40 - (volume/3));
		this.sampler.volume.value = volumeDb;
	}

	startTone() {
		// Returns true if Tone was just started, otherwise returns false (i.e. if Tone had already been started)
		if (!this.toneStarted) { 
			Tone.start().then(() => {
				Tone.Transport.start();
			});
			this.toneStarted = true;
			return true;
		} else {
			return false;
		}
	}
}