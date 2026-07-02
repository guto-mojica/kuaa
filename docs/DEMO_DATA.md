# Demo Data

## Primary Demo Source

The default public demo uses:

- Title: *The Great Train Robbery*
- Year: 1903
- Creator/publisher: Edwin S. Porter / Edison Manufacturing Co.
- Source: Library of Congress item `00694220`
- URL: <https://www.loc.gov/item/00694220/>
- Local filename: `data/demo/runtime/raw/the-great-train-robbery-1903.mp4`

Selection rationale:

- It is short enough for a normal laptop demo.
- It is archival moving-image material rather than modern stock footage.
- It includes visually distinct scenes, interiors/exteriors, people, vehicles,
  and action that exercise the search and annotation workflow.
- The Library of Congress provides a stable public item page and downloadable
  media resources.

Rights note:

The Library of Congress rights statement for this collection says it is not
aware of U.S. copyright or other restrictions for the vast majority of the
motion pictures, while still making users responsible for rights assessment.
This project therefore treats the film as suitable for a public technical demo
with clear attribution, not as legal advice.

Credit line:

Library of Congress, Motion Picture, Broadcasting, and Recorded Sound Division.

## Artifact Policy

The repository does not commit the source video, keyframes, `.npy` embeddings,
or generated runtime metadata. Those files are release artifacts, not source
files.

Tracked in git:

- `data/demo/manifest.json`
- `data/demo/README.md`
- `config/demo.yaml`
- demo docs and validation script

Downloaded/generated locally:

- `data/demo/runtime/raw/`
- `data/demo/runtime/frames/`
- `data/demo/runtime/metadata/`
- `data/demo/runtime/embeddings/`

## Brazilian Regional Context References

The project remains archive-first and Brazilian-context aware. These references
are documented for context and future candidate demos; they are not shipped as
demo artifacts unless redistribution rights are separately verified.

- Cinemateca/BCC: <http://www.bcc.org.br/>
- BCC Mazzaropi collection: <https://www.bcc.org.br/colecoes/mazzaropi>
- BCC `Candinho`: <https://www.bcc.org.br/filmes/891676>
- BCC `Amazonas, O Maior Rio do Mundo`: <https://www.bcc.org.br/filmes/889414>
- Wikimedia Commons `São Paulo, A Sinfonia da Metrópole`:
  <https://commons.wikimedia.org/wiki/File:S%C3%A3o_Paulo,_A_Sinfonia_da_Metr%C3%B3pole_(1929).webm>
- Wikimedia Commons `Ao Redor do Brasil`:
  <https://commons.wikimedia.org/wiki/File:Ao_redor_do_Brasil_(1932)_-_Cinemateca_Brasileira_-_IPHAN.webm>
- Existing development reference: *Jeca Tatu* (1959), Mazzaropi, Internet
  Archive reference linked from the README. Do not redistribute Jeca-derived
  video, keyframes, or artifacts until rights are verified.

## Updating The Demo Bundle

When regenerating a demo bundle:

1. Use `config/demo.yaml`.
2. Keep generated file paths portable under `data/demo/runtime/`.
3. Record model/config identity in the bundle manifest.
4. Update checksums in `data/demo/manifest.json` if the release is final.
5. Run `uv run python scripts/prepare_demo.py --check`.
6. Update screenshots and the demo verification notes with the release tag.
