// Number Counter Animation
const counters = document.querySelectorAll('.counter');
const speed = 100; // lower is slower

counters.forEach(counter => {
    const updateCount = () => {
        const target = +counter.getAttribute('data-target');
        const count = +counter.innerText.replace(/,/g, '');
        const inc = target / speed;

        if (count < target) {
            counter.innerText = (count + inc).toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2});
            setTimeout(updateCount, 20);
        } else {
            counter.innerText = target.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2});
        }
    };
    setTimeout(updateCount, 500); // delay start
});

// Terminal Simulator
const terminalContent = document.getElementById('terminal-content');
const progressBar = document.getElementById('sync-progress');

// Check if realData was loaded from real_data.js
const useRealData = typeof realData !== 'undefined' && realData.length > 0;
let dataQueue = useRealData ? [...realData] : [];
let dummyItems = ['HOBBIES_1_001_CA_1', 'HOUSEHOLD_2_104_TX_2', 'FOODS_3_555_WI_3', 'HOBBIES_2_149_CA_3'];
let processCount = 0;
const totalCount = useRealData ? realData.length : 30; // Max 30 if dummy
let logInterval;

const generateLog = () => {
    // Stop condition
    if (processCount >= totalCount) {
        clearInterval(logInterval);
        const div = document.createElement('div');
        div.className = 'log-line system highlight';
        div.innerHTML = `[${new Date().toISOString().split('T')[1].slice(0, 12)}] [SUCCESS] === Nightly AI-to-ERP Sync Completed ===`;
        terminalContent.appendChild(div);
        
        // Stop the progress bar animation
        progressBar.style.animation = 'none';
        progressBar.style.width = '100%';
        document.querySelector('.sync-status').innerText = 'Synchronization Complete (100%)';
        return;
    }

    const div = document.createElement('div');
    div.classList.add('log-line');
    const timestamp = new Date().toISOString().split('T')[1].slice(0, 12);
    
    // Inject system checks randomly, but only if not the very last few items
    const isSystem = Math.random() > 0.85 && (totalCount - processCount > 5);
    
    if (isSystem) {
        div.classList.add('system');
        div.innerHTML = `[${timestamp}] [INFO] Validating XML-RPC connection stability to Odoo server... OK`;
    } else {
        let item, forecast;
        if (useRealData && dataQueue.length > 0) {
            const dataPoint = dataQueue.shift();
            item = dataPoint.item;
            forecast = dataPoint.forecast;
        } else {
            item = dummyItems[Math.floor(Math.random() * dummyItems.length)];
            forecast = (Math.random() * 15).toFixed(2);
        }
        
        const buffer = Math.floor(forecast * 3 * 1.15);
        div.classList.add('highlight');
        div.innerHTML = `[${timestamp}] [ERP Update] Item: ${item} | Forecast: ${forecast} | Safety Stock: ${buffer}`;
        processCount++;
        
        // Update progress text
        const pct = Math.round((processCount / totalCount) * 100);
        document.querySelector('.sync-status').innerText = `Pushing Safety Stock Triggers... (${pct}%)`;
    }

    terminalContent.appendChild(div);

    // Keep only last 12 lines
    if (terminalContent.children.length > 12) {
        terminalContent.removeChild(terminalContent.firstChild);
    }
};

// Initial logs
setTimeout(() => {
    const div = document.createElement('div');
    div.className = 'log-line system';
    div.innerHTML = `[INFO] Initializing Nightly AI-to-ERP Procurement Pipeline...`;
    terminalContent.appendChild(div);
}, 500);

setTimeout(() => {
    const div = document.createElement('div');
    div.className = 'log-line system';
    div.innerHTML = `[INFO] Triggering the Ultimate Hybrid Meta-Ensemble (Two-Stage Hurdle Model)...`;
    terminalContent.appendChild(div);
}, 1500);

// Start streaming logs
setTimeout(() => {
    logInterval = setInterval(generateLog, 300); // 300ms for faster, dynamic look
}, 2500);
