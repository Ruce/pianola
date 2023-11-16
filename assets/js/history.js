class History {
	constructor(bpm, name) {
		this.bpm = bpm;
		this.name = name;
		this.start = new Date();
		this.noteHistory = [];
		this.lastSeedNoteTime = 0;
		this.parentHistory = null;
		this.isNew = false; // Tracks whether any new (non-rewind) notes have been played
	}
	
	static getRecentHistory(noteHistory, startTime) {
		// Returns events in history that happened on or after `startTime`
		// `noteHistory` must be an ordered list of events from first to most recent
		const recentHistory = [];
		for (let i = noteHistory.length - 1; i >= 0; i--) {
			const h = noteHistory[i];
			if (h.time >= startTime) {
				recentHistory.unshift(h);
			} else {
				break;
			}
		}
		return recentHistory;
	}
	
	static removeHistory(noteHistory, startTime) {
		// Remove events in history that happened on or after `startTime`
		// `noteHistory` must be an ordered list of events from first to most recent
		// Returns new array without altering original `noteHistory`
		for (let i = noteHistory.length - 1; i >= 0; i--) {
			if (noteHistory[i].time < startTime) {
				return noteHistory.slice(0, i+1);
			}
		}
		return [];
	}
	
	add(note) {
		if (!note.isRewind) this.isNew = true;
		
		// Ensure that note is added in chronological order within noteHistory
		let i = this.noteHistory.length - 1;
		while (i >= 0) {
			if (this.noteHistory[i].time <= note.time) {
				break;
			}
			i--;
		}
		this.noteHistory.splice(i+1, 0, note);
	}
	
	copy() {
		// Deep copy this History with new copies of Notes that have isRewind = true
		const newHistory = new History(this.bpm, this.name);
		newHistory.lastSeedNoteTime = this.lastSeedNoteTime;
		newHistory.parentHistory = this;
		for (const note of this.noteHistory) {
			newHistory.add(new Note(note.key, note.velocity, note.duration, note.time, note.actor, null, true));
		}
		return newHistory;
	}
	
	toJsonString() {
		const simplifiedNoteHistory = [];
		for (const note of this.noteHistory) {
			const noteCopy = {
				'k': note.key.keyNum,
				'd': Number(note.duration.toFixed(4)),
				'v': Number(note.velocity.toFixed(4)),
				't': Number(note.time.toFixed(4)),
				'a': note.actor.name
			};
			simplifiedNoteHistory.push(noteCopy);
		}
		
		const simplifiedHistory = { 'id': this.uuid, 'name': this.name, 'bpm': this.bpm, 'noteHistory': simplifiedNoteHistory };
		return JSON.stringify(simplifiedHistory);
	}
}

class HistoryController {
	constructor(endpoint) {
		this.endpoint = endpoint;
		this.allHistories = [];
		this.allPianoRolls = [];
	}
	
	addToHistoryList(history, piano) {
		if (history === null || !history.isNew) return;
		this.allHistories.push(history);
		const historyIdx = this.allHistories.length - 1;
		
		const listContainer = document.getElementById('historyDrawerList');
		const historyElement = document.createElement('li');
		const pianoRoll = new PianoRoll(history);
		this.allPianoRolls.push(pianoRoll);
		
		const textElement = document.createElement('div');
		const heartIcon = document.createElement('i');
		const shareIcon = document.createElement('i');
		historyElement.appendChild(pianoRoll.canvas);
		historyElement.appendChild(textElement);
		historyElement.appendChild(heartIcon);
		historyElement.appendChild(shareIcon);
		historyElement.addEventListener('click', () => piano.replayHistory(historyIdx));
		listContainer.appendChild(historyElement);
		
		// Get the total length of this piece and format to string
		const historyLength = history.noteHistory.at(-1).time + history.noteHistory.at(-1).duration - history.noteHistory[0].time;
		const historySeconds = Math.ceil(historyLength % 60);
		const historyMinutes = Math.floor(historyLength / 60);
		const historyLengthStr = `${historyMinutes}:${historySeconds.toString().padStart(2, '0')}`;
		
		const dateOptions = {day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit', second: '2-digit'};
		textElement.classList.add('historyTextContainer');
		textElement.innerHTML = `<span class="historyTitle">${this.allHistories.length}. ${history.name}</span>`;
		textElement.innerHTML += `<span class="historyDescription">${history.start.toLocaleString('en-US', dateOptions)}</span>`;
		textElement.innerHTML += `<span class="historyDescription">${historyLengthStr}</span>`;
		
		heartIcon.classList.add('heartIcon');
		heartIcon.classList.add('fa-regular');
		heartIcon.classList.add('fa-heart');
		heartIcon.addEventListener('click', toggleHeartIcon);
		
		shareIcon.classList.add('shareIcon');
		shareIcon.classList.add('fa-regular');
		shareIcon.classList.add('fa-share-from-square');
		shareIcon.addEventListener('click', (event) => this.shareHistory(event, historyIdx));
		
		// Check if this history is a variant of another
		let parent = history.parentHistory;
		while (parent !== null) {
			if (this.allHistories.indexOf(parent) !== -1) {
				break;
			} else {
				parent = parent.parentHistory;
			}
		}
		if (parent !== null) {
			textElement.innerHTML += `<span class="historyDescription">Variant of History ${this.allHistories.indexOf(parent) + 1}</span>`;;
		}
		
		pianoRoll.draw();
	}
	
	async shareHistory(event, historyIdx) {
		dismissShareLinkTooltip(event);
		event.stopPropagation();
		const shareLink = await this.createSharedHistoryLink(this.allHistories[historyIdx]);
		
		const linkContainer = document.createElement('div');
		linkContainer.classList.add('shareLinkContainer');
		
		// Set the position of the link container based on the click event
		const linkContainerWidth = 270;
		linkContainer.style.width = linkContainerWidth + 'px';
		linkContainer.style.left = (event.clientX + window.pageXOffset - linkContainerWidth) + 'px';
		linkContainer.style.top = (event.clientY + window.pageYOffset + 14) + 'px';
		
		const linkDescription = document.createElement('span');
		linkDescription.classList.add('shareLinkDescription');
		linkDescription.textContent = 'Share your composition with this link:';
		
		const linkTextElement = document.createElement('input');
		linkTextElement.classList.add('shareLinkInputText');
		linkTextElement.type = 'text';
		linkTextElement.value = shareLink;
		
		const linkCopyIcon = document.createElement('i');
		linkCopyIcon.classList.add('copyIcon');
		linkCopyIcon.classList.add('fa-regular');
		linkCopyIcon.classList.add('fa-copy');
		linkCopyIcon.addEventListener('click', () => navigator.clipboard.writeText(shareLink));
		
		const linkCloseButton = document.createElement('button');
		linkCloseButton.classList.add('closeButton');
		linkCloseButton.addEventListener('click', () => linkContainer.remove());
		
		linkContainer.appendChild(linkDescription);
		linkContainer.appendChild(linkCloseButton);
		linkContainer.appendChild(linkTextElement);
		linkContainer.appendChild(linkCopyIcon);
		document.body.appendChild(linkContainer);
		
		linkTextElement.setSelectionRange(0, linkTextElement.value.length);
		linkTextElement.focus();
	}
	
	async createSharedHistoryLink(history) {
		const uuid = history.uuid ? history.uuid : await this.uploadHistory(history);
		return `${window.location.origin}${window.location.pathname}?${new URLSearchParams({play: uuid})}`;
	}
	
	async uploadHistory(history) {
		function generateRandomId(length) {
			let result = '';
			const characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
			const charactersLength = characters.length;
			for (let i = 0; i < length; i++) {
				result += characters.charAt(Math.floor(Math.random() * charactersLength));
			}
			return result;
		}
		
		// Generate a random id if the history object does not already have one, and PUT the history into the database
		if (!history.uuid) history.uuid = generateRandomId(7);
		
		const requestOptions = {
			method: 'PUT',
			headers: { 'Content-Type': 'application/json' },
			body: history.toJsonString()
		};
		await fetch(this.endpoint, requestOptions);
		return history.uuid;
	}
	
	async getSharedHistory(uuid, piano) {
		const endpointURI = this.endpoint + new URLSearchParams({id: uuid});
		try {
			const response = await fetch(endpointURI, {method: 'GET'});
			const data = await response.json();
			if (!data.Item) return null;
			
			const sharedHistory = new History(data.Item.bpm, data.Item.name);
			sharedHistory.uuid = data.Item.id;
			for (const note of data.Item.noteHistory) {
				sharedHistory.add(new Note(piano.pianoKeys[note.k], note.v, note.d, note.t, Actor.Actors.find(a => a.name === note.a)));
			}
			this.addToHistoryList(sharedHistory, piano);
			return sharedHistory;
		} catch (error) {
			console.log('Error fetching shared history:', error);
		}
	}
}