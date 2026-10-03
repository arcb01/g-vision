'use strict';

const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('gvision', {
  onMessage: (cb) => ipcRenderer.on('gvision:message', (_e, msg) => cb(msg)),
  onConnection: (cb) => ipcRenderer.on('gvision:connection', (_e, state) => cb(state)),
  getConnection: () => ipcRenderer.invoke('gvision:connection'),
  send: (msg) => ipcRenderer.invoke('gvision:send', msg),
});
