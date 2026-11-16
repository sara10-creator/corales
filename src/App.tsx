import { ChangeEvent, useEffect, useRef, useState } from "react";

const coralPhoto =
  "https://images.unsplash.com/photo-1757529151466-6510ee0cf8a4?crop=entropy&cs=tinysrgb&fit=crop&fm=jpg&ixlib=rb-4.1.0&q=85&w=1600";

function SnakeMark({ className = "" }: { className?: string }) {
  return (
    <svg
      className={className}
      viewBox="0 0 320 520"
      role="img"
      aria-label="Ilustración de una serpiente coral mirando hacia abajo"
    >
      <defs>
        <path
          id="snake-path"
          d="M160 18 C68 18 55 108 148 127 C252 148 254 237 158 258 C61 279 64 366 155 383 C204 392 217 424 190 459"
        />
        <clipPath id="snake-clip">
          <use
            href="#snake-path"
            fill="none"
            stroke="white"
            strokeWidth="54"
            strokeLinecap="round"
          />
        </clipPath>
      </defs>

      <use
        href="#snake-path"
        fill="none"
        stroke="#151511"
        strokeWidth="60"
        strokeLinecap="round"
        opacity=".12"
        transform="translate(3 5)"
      />
      <g clipPath="url(#snake-clip)">
        <rect width="320" height="520" fill="#171714" />
        <rect y="42" width="320" height="35" fill="#f0c85a" />
        <rect y="77" width="320" height="62" fill="#c94032" />
        <rect y="139" width="320" height="27" fill="#f0c85a" />
        <rect y="205" width="320" height="35" fill="#f0c85a" />
        <rect y="240" width="320" height="62" fill="#c94032" />
        <rect y="302" width="320" height="27" fill="#f0c85a" />
        <rect y="366" width="320" height="35" fill="#f0c85a" />
        <rect y="401" width="320" height="62" fill="#c94032" />
        <rect y="463" width="320" height="27" fill="#f0c85a" />
      </g>
      <path
        d="M190 437 C218 441 226 466 210 488 C199 503 170 505 158 488 C142 467 158 439 190 437Z"
        fill="#171714"
      />
      <path
        d="M162 468 C151 465 144 458 140 451"
        fill="none"
        stroke="#c94032"
        strokeWidth="4"
        strokeLinecap="round"
      />
      <circle cx="172" cy="477" r="3.5" fill="#f5ead0" />
      <circle cx="202" cy="477" r="3.5" fill="#f5ead0" />
      <circle cx="172" cy="477" r="1.5" fill="#171714" />
      <circle cx="202" cy="477" r="1.5" fill="#171714" />
    </svg>
  );
}

function LeafIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M19.7 4.3C12.8 4.1 7 6.7 5.2 11.2c-1.1 2.8.1 5.2 2.2 6.3 2.4 1.2 5.6.2 7.5-2.1 2.4-2.9 2.7-6.9 4.8-11.1Z" />
      <path d="M4 20c2.6-4.2 6.2-7.1 10.9-9.1" />
    </svg>
  );
}

function CameraIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M4 7.5h3l1.5-2h7l1.5 2h3a2 2 0 0 1 2 2v8.5a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V9.5a2 2 0 0 1 2-2Z" />
      <circle cx="12" cy="13.5" r="4" />
    </svg>
  );
}

function UploadIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 16V3m0 0L7.5 7.5M12 3l4.5 4.5M4 14v5a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-5" />
    </svg>
  );
}

export default function App() {
  const [started, setStarted] = useState(false);
  const [introVisible, setIntroVisible] = useState(true);
  const [preview, setPreview] = useState<string | null>(null);
  const [fileName, setFileName] = useState("");
  const uploadRef = useRef<HTMLInputElement>(null);
  const cameraRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    return () => {
      if (preview) URL.revokeObjectURL(preview);
    };
  }, [preview]);

  const begin = () => {
    setStarted(true);
    window.setTimeout(() => {
      setIntroVisible(false);
      window.scrollTo({ top: 0 });
    }, 1050);
  };

  const selectPhoto = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    if (preview) URL.revokeObjectURL(preview);
    setPreview(URL.createObjectURL(file));
    setFileName(file.name || "Foto tomada");
  };

  const clearPhoto = () => {
    if (preview) URL.revokeObjectURL(preview);
    setPreview(null);
    setFileName("");
    if (uploadRef.current) uploadRef.current.value = "";
    if (cameraRef.current) cameraRef.current.value = "";
  };

  return (
    <div className={`site-shell ${started ? "has-started" : ""}`}>
      {introVisible && (
        <section className={`intro ${started ? "intro-leaving" : ""}`} aria-label="Presentación">
          <div className="intro-copy">
            <span className="eyebrow">Fauna colombiana</span>
            <h1>
              Conoce antes
              <br />
              de temer.
            </h1>
          </div>
          <SnakeMark className="intro-snake" />
          <div className="intro-action">
            <p>Una mirada distinta a la serpiente coral</p>
            <button type="button" onClick={begin}>
              Iniciar
              <span aria-hidden="true">↓</span>
            </button>
          </div>
        </section>
      )}

      <main className="main-content">
        <header className="header">
          <a href="#inicio" className="brand" aria-label="Coral, inicio">
            <span className="brand-mark" aria-hidden="true" />
            CORAL
          </a>
          <nav aria-label="Navegación principal">
            <a href="#conocer">Conocer</a>
            <a href="#observar">Observar</a>
          </nav>
        </header>

        <section className="hero" id="inicio">
          <div className="hero-copy">
            <span className="eyebrow">Respeta lo que no conoces</span>
            <h2>
              No mates
              <br />
              lo desconocido.
            </h2>
            <p>
              El miedo nace de no saber. Observar a distancia y llamar a una persona experta puede
              salvar una vida: la tuya y la de la serpiente.
            </p>
            <a className="text-link" href="#conocer">
              Aprende a convivir <span aria-hidden="true">↘</span>
            </a>
          </div>
          <figure className="hero-photo">
            <img src={coralPhoto} alt="Serpiente coral de bandas rojas, amarillas y negras" />
            <figcaption>
              <span>01</span>
              Las corales son parte vital del equilibrio natural.
            </figcaption>
          </figure>
        </section>

        <section className="info-section" id="conocer">
          <div className="section-heading">
            <span className="eyebrow">Una vecina discreta</span>
            <h2>
              La coral
              <br />
              en Colombia
            </h2>
          </div>
          <div className="info-body">
            <p className="lead">
              En Colombia habitan varias especies del género <em>Micrurus</em>. Sus colores intensos
              advierten, pero su comportamiento suele ser reservado: prefieren huir, ocultarse bajo
              la hojarasca o permanecer bajo tierra.
            </p>
            <div className="facts">
              <article>
                <span className="fact-icon">
                  <LeafIcon />
                </span>
                <div>
                  <h3>Habita sin hacer ruido</h3>
                  <p>
                    Se encuentra en bosques, zonas rurales y suelos con abundante cobertura vegetal.
                  </p>
                </div>
              </article>
              <article>
                <span className="fact-number">01</span>
                <div>
                  <h3>Mírala, no la manipules</h3>
                  <p>
                    No intentes capturarla ni acercarte. Mantén distancia y permite que siga su
                    camino.
                  </p>
                </div>
              </article>
              <article>
                <span className="fact-number">02</span>
                <div>
                  <h3>Si hay una mordedura</h3>
                  <p>Busca atención médica inmediata. No hagas torniquetes, cortes ni succiones.</p>
                </div>
              </article>
            </div>
          </div>
        </section>

        <aside className="care-note">
          <p>La mayoría de los encuentros terminan bien cuando damos espacio.</p>
          <span>Distancia · Calma · Respeto</span>
        </aside>

        <section className="observe-section" id="observar">
          <div className="observe-copy">
            <span className="eyebrow">Registro responsable</span>
            <h2>
              ¿Viste una
              <br />
              serpiente?
            </h2>
            <p>
              Toma una foto únicamente desde una distancia segura. Puedes revisarla aquí sin que
              salga de tu dispositivo.
            </p>
          </div>

          <div className={`photo-panel ${preview ? "has-preview" : ""}`}>
            {preview ? (
              <div className="preview-wrap">
                <img src={preview} alt="Vista previa de la fotografía seleccionada" />
                <div className="preview-meta">
                  <div>
                    <span>Foto lista</span>
                    <strong>{fileName}</strong>
                  </div>
                  <button type="button" onClick={clearPhoto}>
                    Cambiar foto
                  </button>
                </div>
              </div>
            ) : (
              <>
                <div className="panel-symbol">
                  <CameraIcon />
                </div>
                <div>
                  <h3>Sube una foto o usa tu cámara</h3>
                  <p>Formatos JPG, PNG o HEIC. No te acerques para conseguir una mejor imagen.</p>
                </div>
                <div className="photo-actions">
                  <button
                    type="button"
                    className="primary-action"
                    onClick={() => uploadRef.current?.click()}
                  >
                    <UploadIcon />
                    Elegir foto
                  </button>
                  <button
                    type="button"
                    className="secondary-action"
                    onClick={() => cameraRef.current?.click()}
                  >
                    <CameraIcon />
                    Tomar foto
                  </button>
                </div>
              </>
            )}
            <input
              ref={uploadRef}
              className="visually-hidden"
              type="file"
              accept="image/*"
              onChange={selectPhoto}
              aria-label="Elegir una fotografía"
            />
            <input
              ref={cameraRef}
              className="visually-hidden"
              type="file"
              accept="image/*"
              capture="environment"
              onChange={selectPhoto}
              aria-label="Tomar una fotografía con la cámara"
            />
          </div>
          <p className="safety-caption">
            Esta herramienta solo muestra una vista previa y no identifica especies. Ante una
            emergencia, comunícate con la línea 123.
          </p>
        </section>

        <footer>
          <a href="#inicio" className="brand">
            <span className="brand-mark" aria-hidden="true" />
            CORAL
          </a>
          <p>Conocer también es cuidar.</p>
          <a href="#inicio">Volver arriba ↑</a>
        </footer>
      </main>
    </div>
  );
}
