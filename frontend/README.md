# The viewer has moved

The single viewer source now lives in `mechlens-cloud/frontend/src/viewer`.

For standalone development, run `npm ci` then `npm run dev:local` in
`mechlens-cloud/frontend`, and `mechlens serve` on the GPU machine.
Open http://localhost:5173; the default API is http://localhost:8000.
Set `VITE_API_BASE_URL` to override the local API URL.

`make dev` in Mechlens uses that same Cloud checkout (override its location
with `CLOUD_FRONTEND=/path/to/mechlens-cloud/frontend`).

For the hosted workspace, run `make up` in `mechlens-cloud` and pair the GPU
with `mechlens serve --cloud http://localhost:5175`.
