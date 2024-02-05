class PianolaMidi {
	constructor(piano) {
		this.piano = piano;
		this.midiAccess = null;
		this.table = document.getElementById('midiDeviceTable');
		this.initialiseMidi();
	}
	
	static get tablePlaceholderMessage() {
		return '[No MIDI devices detected]';
	}
	
	static getEnableCheckboxElement(row) {
		for (const el of row.cells) {
			if (el.classList.contains('midiDeviceEnable')) {
				return el.querySelector('input');
			}
		}
	}
	
	static getDeviceLabel(row) {
		for (const el of row.cells) {
			if (el.classList.contains('midiDeviceName')) {
				return el.querySelector('label');
			}
		}
	}
	
	initialiseMidi() {
		// Add placeholder row to settings table
		this.addTablePlaceholder();
		
		navigator.requestMIDIAccess().then((access) => {
			this.midiAccess = access;
			this.populateMidiTable();
			
			// If a new midi port is connected, attach an onmessage listener and update settings table
			this.midiAccess.onstatechange = (event) => {
				this.updateMidiTable(event);
			}
		});
	}
	
	attachMidiListener(input) {
		if (!input.onmidimessage) {
			input.onmidimessage = (message) => {
				this.piano.onMidiMessage(message);
			};
		}
	}
	
	detachMidiListener(input) {
		input.onmidimessage = null;
	}
	
	createTableRow(tBody, id, deviceName, isChecked, isDisabled, input=null) {
		/*
		<td class="midiDeviceEnable"><label class="checkboxLabel"><input id="placeholder" name="enable" type="checkbox" disabled></label></td>
		<td class="midiDeviceName"><label for="placeholder">[No MIDI devices detected]</label></td>
		*/
		
		const newRow = tBody.insertRow();
		
		const enableCell = newRow.insertCell(0);
		enableCell.classList.add('midiDeviceEnable');
		
		const enableLabel = document.createElement('label');
		enableLabel.classList.add('checkboxLabel');
		enableCell.appendChild(enableLabel);
		
		const enableInput = document.createElement('input');
		enableInput.id = id;
		enableInput.setAttribute('name', 'enable');
		enableInput.setAttribute('type', 'checkbox');
		enableInput.checked = isChecked;
		enableInput.disabled = isDisabled;
		if (input) {
			enableInput.addEventListener('change', () => {
				if (enableInput.checked) {
					this.attachMidiListener(input);
				} else {
					this.detachMidiListener(input);
				}
			});
		}	
		enableLabel.appendChild(enableInput);
		
		const nameCell = newRow.insertCell(1);
		nameCell.classList.add('midiDeviceName');
		
		const nameLabel = document.createElement('label');
		nameLabel.setAttribute('for', id);
		nameLabel.textContent = deviceName;
		nameCell.appendChild(nameLabel);
	}
	
	addTablePlaceholder() {
		this.createTableRow(this.table.tBodies[0], 'midiDevicePlaceholder', PianolaMidi.tablePlaceholderMessage, false, true);
	}
	
	getCheckboxElementId(input) {
		const idPrefix = 'midiDevice';
		return `${idPrefix}${input.id}`;
	}
	
	addDevice(input) {
		// Add device to settings table and attach listener
		this.attachMidiListener(input);
		
		const tBody = this.table.tBodies[0];
		if (tBody.rows.length === 1) {
			// Check if the first row is the default placeholder or an actual device
			const deviceEnable = PianolaMidi.getEnableCheckboxElement(tBody.rows[0]);
			if (deviceEnable.disabled) {
				// Is a placeholder, remove the row
				this.table.deleteRow(tBody.rows[0].rowIndex);
			}
		}
		
		// Add a new row to table
		this.createTableRow(tBody, this.getCheckboxElementId(input), input.name, true, false, input);
	}
	
	removeDevice(input) {
		// Remove device from table and detach listener
		this.detachMidiListener(input);
		
		// Find the table row with this device and delete it from the table
		const tBody = this.table.tBodies[0];
		const checkboxElementId = this.getCheckboxElementId(input);
		for (const row of tBody.rows) {
			const deviceEnable = PianolaMidi.getEnableCheckboxElement(row);
			if (deviceEnable && deviceEnable.id === checkboxElementId) {
				// Row found
				this.table.deleteRow(row.rowIndex);
				
				// If there are no rows left in the table, add placeholder
				if (tBody.rows.length === 0) this.addTablePlaceholder();
				
				break;
			}
		}
	}
	
	populateMidiTable() {
		this.midiAccess.inputs.forEach((input) => {
			this.addDevice(input);
		});
	}
	
	updateMidiTable(event) {
		if (!event.port || event.port.type !== 'input') return;
		
		if (event.port.state === 'connected') {
			// Check if the device is already listed in the settings table
			// There can be multiple connection events for the same device and the state info is unreliable (as they are updated async)
			const deviceEnable = document.getElementById(this.getCheckboxElementId(event.port));
			if (!deviceEnable) this.addDevice(event.port);
		} else if (event.port.state === 'disconnected') {
			this.removeDevice(event.port);
		}
	}
}
