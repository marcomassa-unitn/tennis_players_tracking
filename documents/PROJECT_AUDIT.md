# Audit critico del progetto — Tennis Players Tracking

> ## ✅ STATO (2026-06-19): tutti i punti risolti
>
> Risolti in una passata di **hardening multi-agente** (5 agenti su gruppi di
> file disgiunti) + verifica end-to-end. In sintesi:
> - **CRITICAL**: C1 (`CourtTracker` ora onora `output_dir`), C2 (ogni step della
>   pipeline è isolato in `try/except (Exception, SystemExit)`), C3 (`step_shots`
>   salta con grazia se manca il CSV palla), C4 (`try/finally` + release di
>   capture/writer in tutti i `run()`).
> - **HIGH**: H1 (no-lines → `RuntimeError`, niente blocco headless), H2 (pipeline
>   headless di default + `--display`), H3 (`--flow-frames`, default 200), H4
>   (guard NaN su velocità medie), H5/H6 (gradient su run contigui + dedup min-gap
>   corretto), H7 (soglie tiri esposte via CLI), H8 (guard FPS unico `isfinite`),
>   H9 (area minima componenti + fallback con `MAX_MOVE`), H10 (`--court` passato
>   a `evaluate_tracking`).
> - **MEDIUM/LOW**: clustering deterministico, RANSAC (court + converter), guard
>   orizzonte, lettura sequenziale block-matching, gating IoU nelle metriche,
>   `clip2` case-insensitive, FPS reale letto dal video, colori P1/P2 coerenti,
>   `frames_missing` corretto, costanti nominate al posto dei numeri magici, ecc.
> - **Feature nuova**: rilevamento/correzione **swap ID** P1↔P2 (assegnamento 2×2
>   a costo minimo con gating `MAX_MOVE`) in `playerTracking.py`.
>
> **Verifiche**: tutti gli 11 file compilano; pipeline completa headless OK;
> self-test tiri 4/4; C1 e C3 verificati a runtime.
>
> Il testo sottostante è l'audit diagnostico **originale** (pre-fix), conservato
> come riferimento.

Analisi critica approfondita di tutti i moduli `.py` e di `pipeline.py`.
Per ogni voce: `file:riga`, gravità, descrizione e perché conta.
Le voci segnate **(verificato)** sono state confermate eseguendo/leggendo il
codice, non solo dedotte.

**Legenda gravità:** CRITICAL = rompe il funzionamento / corruzione dati ·
HIGH = bug reale con impatto frequente · MEDIUM = bug in casi non rari /
metrica fuorviante · LOW/NIT = robustezza, stile, debito tecnico.

---

## Verdetto sintetico

Gli algoritmi *centrali* sono nel complesso corretti (verificato:
l'omografia px→metri è accurata al mm, la conversione km/h `*3.6` è giusta, le
dimensioni ITF del campo sono corrette). I problemi gravi NON sono nella
matematica ma in **integrazione, gestione risorse, gestione errori e
robustezza** — cioè proprio nei punti che trasformano "script che girano da
soli" in "pipeline unica affidabile".

I quattro nodi trasversali:
1. **Nessun `try/finally`**: ogni `VideoCapture`/`VideoWriter` viene perso su
   qualsiasi eccezione (in tutti e tre i tracker).
2. **Nessun isolamento d'errore per-step nella pipeline**: un `SystemExit` di
   un sotto-modulo abbatte l'intera pipeline.
3. **Path di output incoerenti**: `CourtTracker` scrive su `outputs/` hardcoded,
   ignorando `--output`; il resto della pipeline legge da `--output`.
4. **Soglie "magiche" tarate su un solo clip 1080p**: non generalizzano ad altri
   video (fps/risoluzione diversi).

---

## CRITICAL

### C1 — `CourtTracker` scrive su un path hardcoded, ignorando `--output` (verificato)
`court_tracking.py:324` · `pipeline.py:70`
```python
# court_tracking.py:324
csv_path = os.path.join("outputs", "court_coordinates", f"{video_stem}_court.csv")
```
`CourtTracker.__init__` non ha alcun parametro di output. Con un `--output`
diverso da `outputs/`, `derive_paths` calcola `court_csv = <output>/court_coordinates/...`
ma il CSV finisce comunque in `outputs/...`. Tutti gli step a valle (3/4/6/7)
ricevono `--court <output>/...` → file inesistente → `FileNotFoundError` /
`CourtConverter` solleva → pipeline interrotta. È anche relativo alla CWD: se
lanci da un'altra cartella, il file finisce nel posto sbagliato in silenzio.

### C2 — Nessun isolamento d'errore per-step nella pipeline (verificato)
`pipeline.py` (tutti gli `step_*`) · `_override_argv` a `pipeline.py:36`
`_override_argv` ripristina `sys.argv` correttamente (ha `try/finally`) ma **non
cattura le eccezioni**: un `SystemExit` o qualunque errore di un `main()`
pilotato propaga e uccide tutto, scartando gli step successivi indipendenti.
Inneschi reali: vedi C3 e H1. Manca un `try/except` attorno a ogni step con
log "step fallito, continuo".

### C3 — `step_shots` non verifica il prerequisito CSV palla → crash dell'intera pipeline (verificato)
`pipeline.py:235` · `shot_analysis.py:430`
```python
# shot_analysis.py:430
if not os.path.exists(args.ball):
    raise SystemExit("Ball CSV not found: ...")
```
`step_ball` salta legittimamente quando mancano `ultralytics` o i pesi
`ball_tracker.pt` (caso molto comune). Ma `step_shots` controlla solo
`players_csv`, non `ball_csv`, e passa `--ball <path_inesistente>` →
`shot_analysis.main()` fa `raise SystemExit` → **tutta la pipeline muore allo
step 6**, anche se court/tracking/analisi/motion erano andati a buon fine. Questo
ti colpirà al primo run senza pesi YOLO.

### C4 — Leak di `VideoCapture`/`VideoWriter` su ogni percorso non "felice" (verificato)
`court_tracking.py:332+` · `playerTracking.py:168-205` · `BallTracking.py:81-142`
Nessun `run()` usa `try/finally`. Qualsiasi eccezione nel loop (es. `cvtColor`
su un frame corrotto, errore del modello YOLO) salta `cap.release()` /
`writer.release()`. Conseguenze: handle del file/dispositivo persi (su Windows il
file resta lockato) e, per il writer, **file `.mp4` di output corrotto**
(container non finalizzato). In `PlayerTracker` anche il CSV resta aperto e
troncato. È il bug più pervasivo del progetto.

---

## HIGH

### H1 — Percorso "nessuna linea" del court: blocca in headless e poi `raise SystemExit` (verificato)
`court_tracking.py:371-376`
```python
if raw_lines is None:
    print("No lines detected — ...")
    cv2.imshow("White Mask", ...)
    cv2.waitKey(0)        # blocca all'infinito anche se no_display=True
    cv2.destroyAllWindows()
    raise SystemExit       # uccide il processo, non catchabile come eccezione
```
Non consulta `self.no_display`: in pipeline (che forza `no_display=True`) si
blocca comunque su un tasto, e poi `SystemExit` abbatte tutto. Non capita sul tuo
video attuale (le linee si trovano) ma è un crash/hang latente.

### H2 — Headless non garantito: `python pipeline.py` senza `--no-display` apre finestre bloccanti (verificato)
`pipeline.py:122` (player) · `pipeline.py:224` (ball)
`display=not args.no_display`: con il default (senza `--no-display`) gli step 2 e
5 aprono finestre OpenCV interattive e **richiedono di premere `q`** per
proseguire. Il docstring consiglia `--no-display` ma nulla lo impone; lo step
court invece è già forzato headless. Incoerenza.

### H3 — Optical flow pilotato senza `--frames`: Farneback denso su tutto il video ad ogni run
`pipeline.py:168-175`
`optical_flow.main()` ha default `--frames 0` = intero video. La pipeline non lo
limita e non passa `--no-lk`, quindi lo step 4 fa flusso ottico denso su migliaia
di frame: può dominare il tempo totale. Serve un cap (es. `--frames`) o un flag.

### H4 — Velocità media/mediana non protette da array tutto-NaN → NaN nell'output (verificato)
`player_analysis.py:166-170`
`avg_speed_kmh`/`median_speed_kmh` usano `np.nanmean`/`np.nanmedian` senza il
guard a `0.0` che invece hanno `max` e `p95`. Una traccia di un solo frame (o
tutta a gap) produce array vuoto/NaN → RuntimeWarning e **NaN** in stat e CSV.
Incoerente con le altre metriche.

### H5 — `shot_analysis`: `np.gradient` su array bucato da NaN contamina i vicini → tiri persi (verificato)
`shot_analysis.py:126-128`
`np.gradient(np.where(valid, cx, np.nan))` propaga NaN a `i±1` di ogni buco: i
test `flip`/`acc_peak` su quegli indici falliscono in silenzio, e tiri reali
vicino a un frame mancante della palla vengono mancati. Va derivato solo su run
contigui validi.

### H6 — `shot_analysis`: dedup `min-gap` si ri-ancora al candidato sostituito → rally collassano in un tiro (verificato)
`shot_analysis.py:164-171`
La finestra di soppressione si misura su `hits[-1][0]`, ma quando `hits[-1]` viene
sovrascritto da un candidato più forte e più tardo, la finestra avanza. Una
sequenza di colpi ciascuno < min_gap ma estesa ben oltre min_gap totale si fonde
in un solo tiro → conteggio tiri sottostimato negli scambi veloci.

### H7 — Soglie di shot-detection sono numeri magici in pixel/frame, tarati su un solo clip
`shot_analysis.py:111-113`
`vy_min=0.5`, `acc_thr=1.5`, `win=4` sono in px/frame e px/frame²: scalano con la
risoluzione e con quanto appare grande il campo. Su un altro video (fps diverso,
4K vs 720p, inquadratura più larga) sovra/sotto-scattano. Non esposti via CLI, e
il self-test usa la stessa geometria hardcoded → non può rilevare il problema.

### H8 — Gestione NaN-FPS incoerente tra i tre tracker (verificato)
`court_tracking.py:468` (`np.isnan` ✓) · `playerTracking.py:31-32` (solo `<=0`) ·
`BallTracking.py:87` (`get(...) or 30.0` → NaN passa, perché NaN è "truthy")
Stesso pattern, tre comportamenti diversi. Con FPS NaN: `PlayerTracker` fa
`int(nan)` → `ValueError`; `BallTracker` crea il `VideoWriter` con `fps=nan` →
video rotto. Servirebbe un guard unico `if fps <= 0 or not np.isfinite(fps): fps = 30`.

### H9 — `PlayerTracker`: nessun filtro di area minima; il primo frame si "semina" dai 2 blob più grandi (verificato)
`playerTracking.py:100-117`
`_find_components` tiene *ogni* componente connessa (solo `area`, nessuna soglia).
Quando `prev_centroids is None` prende i 2 blob più grandi qualunque essi siano
(rete, ombre, tabellone, folla) → seme errato che si propaga lungo tutta la catena
nearest-centroid. Inoltre il fallback a `playerTracking.py:138-144` ignora
`MAX_MOVE`, quindi un blob lontano può essere "teletrasportato" come giocatore.

### H10 — `evaluate_tracking`: la pipeline non passa mai `--court` → `feet_err_m` sempre vuota (verificato)
`pipeline.py:275-281` · `evaluate_tracking.py:221,236`
Senza `--court`, `evaluate_players` gira con `conv=None` e la colonna metrica
dell'errore ai piedi (una metrica di punta pubblicizzata) è sempre vuota in
modalità pipeline.

---

## MEDIUM

- **`pipeline.py:63` — special-case `clip2` fragile e case-sensitive.**
  `key = "clip2" if stem == "Input_video2" else stem.lower()`: `INPUT_VIDEO2.mp4`,
  `input_video2.mp4`, o un nome con spazio finale cadono nel ramo `stem.lower()` e
  NON combaciano con i default documentati dei moduli standalone.
- **`playerTracking.py:35,92` — `THRESH = int(0.5*fps)` confonde fps con soglia di intensità** (0-255) passata a `cv2.threshold`. La sensibilità del foreground cambia con il frame rate senza motivo fisico.
- **`BallTracking.py:44-50` — gap finali restano NaN → perdita dati silenziosa.** `df.interpolate()` non riempie i NaN di coda (default forward); le rilevazioni dopo l'ultima valida spariscono dal CSV.
- **`BallTracking.py:122` — Pass-2 indicizza `ball_positions[idx]` assumendo che i due passaggi leggano lo stesso numero di frame.** Su input VFR i due `VideoCapture` possono disallinearsi → annotazione desincronizzata dalla detection.
- **`court_tracking.py:80-93` — `cluster_points` dipende dall'ordine** (merge nel primo cluster entro raggio + media a coppie): due keypoint distinti entro `CLUSTER_RADIUS=50px` possono fondersi; risultato non deterministico rispetto all'ordine dei segmenti Hough.
- **`court_tracking.py:448-450` — `court_span = bl[1]-tl[1]` può essere ≤0** con omografia degenere → il filtro dei dot si comporta in modo opposto all'intento, senza guard.
- **`player_analysis.py:184,190-193` — confini delle zone asimmetrici** (un giocatore esattamente su una linea è assegnato in modo incoerente) e cutoff out-of-range (`+1.0 m`) diverso dall'estensione di heatmap/zone.
- **`player_analysis.py:223` — `_compute_zone_stats` solleva su input vuoto** (`.sort_values(["player_id","percent"])` su DataFrame senza colonne) invece di restituire stat vuote.
- **`shot_analysis.py:67-69` — `reindex(range(min,max+1))` assume la stessa numerazione di frame tra CSV palla e CSV giocatori.** Se la palla parte da frame 0 e i giocatori da 30, ogni candidato viene scartato → "No shots detected" senza diagnostica.
- **`shot_analysis.py:206-212` — `np.nanmedian` su finestra tutto-NaN → `ball_cx` NaN** passato non protetto a `classify_stroke` (restituisce "backhand" deterministico ma insensato) e a `to_meters` (coordinate NaN).
- **`block_matching.py:227-229` — `cap.set(POS_FRAMES,i); read(); read()` per coppie consecutive è inaffidabile** su molti codec (seek a non-keyframe) → PSNR potenzialmente sbagliato in silenzio.
- **`evaluate_tracking.py:126-127` — il matching può accoppiare box con IoU=0** (best-of-bad): quelle coppie inquinano media/mediana di center/feet error, mescolando errore di localizzazione ed errore di associazione.
- **`evaluate_tracking.py:130-132` — il conteggio "ID switch" misura l'instabilità del match, non la continuità ID del tracker** → sovrastima.
- **`pipeline.py:300` — `--fps` globale fisso a 30.0**: la catena di analisi non legge mai l'fps reale dal video; se il sorgente è 25/60 fps tutte le velocità/tempi sono sbagliati (court/player tracker invece leggono l'fps vero → incoerenza interna).
- **Geometria del campo definita 3 volte:** `REAL_FT` (`court_tracking.py`), `_REAL_WORLD` (`court_converter.py`) e di nuovo importata in `evaluate_tracking.py` → rischio di drift.

---

## LOW / NIT (sintesi)

- **`court_converter.py:81` / `court_tracking.py` — omografia con solve esatto a 4 punti (`method=0`), niente RANSAC**: un solo keypoint rumoroso deforma tutte le coordinate a valle (ci sono 8 corner disponibili per essere robusti).
- **`court_converter.py:39` — nessun guard su `res[:,2]≈0`** (punto sulla linea d'orizzonte) → ±inf metri propagati senza filtro.
- **`player_analysis.py` — colori giocatore incoerenti tra le 3 figure** (P1 = lime / hot / red; P2 = tomato / Blues_r / steelblue): la legenda non mappa in modo affidabile colore→giocatore.
- **`player_analysis.py:175` — `frames_missing` conta gli *eventi* di gap, non i frame mancanti** (un buco da 50 frame conta 1) → metrica fuorviante.
- **Dead code:** `BallTracking.py:35-42` (`detect_frames` mai usato dal CLI), `shot_analysis.py` `self_test` mai esercitato dalla pipeline, vari `import` solo per self-test.
- **`sys.path.insert` hack** duplicato in `player_analysis.py:43`, `shot_analysis.py:47`, `optical_flow.py:40`, `evaluate_tracking.py:41`: fragile; preferibile `python -m` / package installabile.
- **`pipeline.py:212` — `ball_tracker.pt` hardcoded**, nessun passthrough `--model` benché `BallTracker` lo supporti.
- **`pipeline.py:329` — `--anchor` senza `choices=`**: un typo (`foot`) viene inoltrato e fa crashare `player_analysis` invece di essere bloccato subito da argparse.
- **`pipeline.py:362-367` — il dict `produced` non verifica il successo** dei `main()` pilotati: dichiara artefatti che potrebbe non aver prodotto.
- **Numeri magici sparsi** non documentati: `conf=0.65` (`BallTracking.py:27`), `cv2.threshold(...,15,...)` (`BallTracking.py:18`), `delay=int(500/fps)` (`court_tracking.py:470`), `DISPLAY_W=1280` (`annotate.py:36`).

---

## Temi trasversali (le 6 cose da sistemare prima)

1. **Avvolgere ogni `run()` in `try/finally`** che rilascia capture/writer (e chiude il CSV in `PlayerTracker`). → risolve C4.
2. **Isolare gli step della pipeline** con `try/except` + log, e dare a `step_shots` un guard sull'esistenza del CSV palla. → risolve C2, C3.
3. **Unificare il path del court**: o `CourtTracker` accetta/onora una dir di output, o la pipeline legge dal path hardcoded. → risolve C1.
4. **Guard FPS unico** `fps <= 0 or not np.isfinite(fps)` nei tre tracker. → risolve H8.
5. **Esporre le soglie** di shot-detection (e idealmente normalizzarle alla scala del campo) invece di hardcodarle. → mitiga H7.
6. **Rilevare/correggere lo swap di identità** dei giocatori (oggi assunto stabile per tutto il clip; il clamp a 45 km/h ne nasconde solo il sintomo di velocità) → mitiga un'intera classe di errori in analisi/tiri/valutazione.

---

*Nota: questo audit è solo diagnostico — nessun file è stato modificato.*
