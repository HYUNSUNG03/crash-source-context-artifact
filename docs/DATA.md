# Data and source provenance

Reports and bug labels originate from the [GPTrace evaluation artifacts](https://github.com/Fraunhofer-AISEC/gptrace-artifacts). The exact selected cohort is `data/reports.json`: FreeType char2svg 51 reports/6 bugs; Poppler pdfimages 40/4; SoX MP3 61/7. Labels are evaluation-only and are not embedding components.

Source document identifiers in the experiment are FreeType 2.5.3, SoX 14.4.2 and Poppler 1d23101c with Magma v1.2.1 bug patches. See the source keys in `reports.json` and the Poppler reconstruction script in `historical/earlier/`. These identify the recorded source documents; they do not prove a byte-identical rebuild of the crashing binary.

GPTrace evaluation code is pinned at `ecf52c76a07c13d8aa3de71860a5c5d74e6f009f`. Local B differs from GPTrace's original OpenAI/64-dimensional configuration. The comparison is between four representations in one local environment, not between published tool scores.

Only derived numeric data and minimal cohort metadata are bundled. Obtain upstream projects and benchmark artifacts under their respective licenses; their source trees, crash executables, crash-triggering input files, PDFs and model weights are not relicensed or redistributed by this package. The source equality baseline uses the first source-mapped frame's `(source_id, source_relative_path, line)`; this is not a raw top-frame stack hash.
