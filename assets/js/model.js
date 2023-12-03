class PianolaModel {
	constructor(endpoint) {
		this.endpoint = endpoint;
		this.lastActivity = null;
		this.lastQuery = null;
		this.isConnected = false;
		this.keepAliveIntervalId = setInterval(() => this.keepAlive(), 1000);
	}
	
	static get Base52Mapping() {
		return ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'];
	}
	
	static toBase52(number) {
		function decimalToCustomBase(number, targetBase, strMapping){
			let customNumber = "";
			while (number > 0) {
				const remainder = number % targetBase;
				customNumber = strMapping[remainder] + customNumber;
				number = Math.floor(number / targetBase);
			}
			return customNumber;
		}
		return decimalToCustomBase(number, 52, PianolaModel.Base52Mapping);
	}
	
	static fromBase52(numberStr) {
		function customBaseToDecimal(numberStr, sourceBase, strMapping) {
			let total = 0;
			for (const [i, c] of numberStr.split('').reverse().entries()) {
				const value = strMapping.indexOf(c);
				total += parseInt(value * Math.pow(sourceBase, i));
			}
			return total;
		}
		return customBaseToDecimal(numberStr, 52, PianolaModel.Base52Mapping);
	}
	
	static padNum(number, digits) {
		return number.toString().padStart(digits, '0');
	}
	
	static queryStringToNotes(data, start, interval) {
		/*
		Arguments:
			`data`: a String containing the encoded notes information
			`start`: the TransportTime (in seconds) that the first tick should start at
			`interval`: length (in seconds) of a tick interval
		*/
		
		const notesSlices = data.split(/\d+/).filter((x) => x !== '');
		const spaces = Array.from(data.match(/\d+/g), (x) => parseInt(x));

		const generated = [];
		let t = start;
		for (let i=0; i < notesSlices.length; i++) {
			t += spaces[i] * interval;
			const notesStr = notesSlices[i];
			for (let i = 0; i < notesStr.length; i += 4) {
				const noteNumerals = PianolaModel.padNum(PianolaModel.fromBase52(notesStr.slice(i, i + 4)), 7);
				const noteNum = parseInt(noteNumerals.slice(0, 2)) - 2; // N.B. Important: the note was shifted up by 2 when converting to a note string
				const velocity = (parseInt(noteNumerals.slice(2, 4)) + 1) / 100;
				const duration = parseInt(noteNumerals.slice(4, 7)) * interval / 10;
				generated.push({keyNum: noteNum, velocity: velocity, duration: duration, time: t});
			}
			t += interval;
		}
		return generated;
	}
	
	static historyToQueryString(noteHistory, start, end, interval) {
		/*
		Arguments:
			`noteHistory`: an array containing recent history of Notes that were played
			`start`: the TransportTime (in seconds) for the start of history period
			`end`: the TransportTime (in seconds) for the non-inclusive end of history period
			`interval`: length (in seconds) of a tick interval
		*/
		
		const numSixteenthNotes = Math.ceil((end - start) / interval);
		const orderedNotes = Array.from({ length: numSixteenthNotes }, () => []);

		for (const n of noteHistory) {
			if (n.time < end) {
				const t = Math.floor((n.time - start) / interval); // Delta between note and startTick in SixteenthNotes
				const velocity = Math.max(Math.min(Math.round((n.velocity * 100) - 1), 99), 0); // Velocity scaled to be 0-indexed, between 0 and 99
				const duration = Math.max(Math.min(Math.round(n.duration / interval * 10), 999), 1);
				
				// N.B. Important: we shift the note up by 2 so that the base52 string is guaranteed to be 4 characters long
				const noteShifted = n.key.keyNum + 2
				const numberStr = `${PianolaModel.padNum(noteShifted, 2)}${PianolaModel.padNum(velocity, 2)}${PianolaModel.padNum(duration, 3)}`;
				orderedNotes[t].push(PianolaModel.toBase52(numberStr));
			}
		}
		
		let queryString = '';
		let spaces = 0;
		for (const notes of orderedNotes) {
			if (notes.length === 0) {
				spaces++;
			} else {
				// Add the number of spaces before this timestep of active notes
				queryString += spaces.toString() + notes.join('');
				spaces = 0;
			}
		}
		if (spaces > 0) queryString += spaces.toString();
		return queryString;
	}
	
	static fetchWithRetry(url, maxRetries, delayMs, retryCount=0) {
		return fetch(url)
		.then(response => { return response.json(); })
		.catch(error => {
			if (retryCount < maxRetries) {
				return new Promise(resolve => {
					setTimeout(() => { resolve(PianolaModel.fetchWithRetry(url, maxRetries, delayMs, retryCount + 1)); }, delayMs);
				});
			} else {
				throw error;
			}
		});
	}
	
	async queryModel(queryString, timesteps, numRepeats, selectionIdx) {
		this.lastQuery = new Date();
		const endpointURI = this.endpoint + new URLSearchParams({notes: queryString, timesteps: timesteps, num_repeats: numRepeats, selection_idx: selectionIdx});
		const maxRetries = 20;
		const delayMs = 100;
		try {
			const data = await PianolaModel.fetchWithRetry(endpointURI, maxRetries, delayMs);
			return data;
		} catch (error) {
			console.log('Error connecting to endpoint:', error);
		}
	}
	
	async connectToModel(callback) {
		this.lastActivity = new Date();
		const data = await this.queryModel("1", 1, 1, 0);
		console.log(new Date().toISOString(), `Connected to model [${data}]`);
		
		if (typeof data !== 'undefined' && !data.hasOwnProperty('message')) {
			// Expected response received, trigger callback
			this.isConnected = true;
			callback();
		} else {
			// Wait a short while before trying to connect to model again
			setTimeout(this.connectToModel.bind(this, callback), 500);
		}
	}
	
	async generateNotes(history, start, end, interval, timesteps, numRepeats, selectionIdx) {
		/*
		Arguments:
			`history`: an array containing recent history of Notes that were played
			`start`: the TransportTime (in seconds) for the start of history period
			`end`: the TransportTime (in seconds) for the end of history period
			`interval`: length (in seconds) of a tick interval
			`timesteps`: number of timesteps to generate
			`numRepeats`: number of samples to generate from the same seed
			`selectionIdx`: index of the sample to select if generation is repeated
		*/
		this.lastActivity = new Date();
		var generated = [];
		
		if (this.isConnected) {
			const queryString = PianolaModel.historyToQueryString(history, start, end, interval);
			console.log(new Date().toISOString(), 'Query:', queryString);
			const data = await this.queryModel(queryString, timesteps, numRepeats, selectionIdx);
			console.log(new Date().toISOString(), 'Data:', data);
			
			if (!data.hasOwnProperty('message')) {
				const offset = interval / 2; // Half of a tick interval length
				const generatedStart = end + offset; // Generated notes start at an offset from the end time so that they are centered within an interval
				generated = PianolaModel.queryStringToNotes(data, generatedStart, interval);
			}
		}
		return generated;
	}
	
	async keepAlive() {
		const currTime = new Date();
		const activityTimeout = 300000;
		const queryInterval = 20000;
		// If the model has been active in `activityTimeout` window and no queries have been made in the last `queryInterval`, query the model to keep it alive
		if (this.lastActivity !== null && currTime - this.lastActivity < activityTimeout && this.lastQuery !== null && currTime - this.lastQuery > queryInterval) {
			const data = await this.queryModel("1", 1, 1, 0);
		}
	}
}
