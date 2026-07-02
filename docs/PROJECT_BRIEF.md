# Project brief

## Summary

KUAA is an offline multimodal AI tool for turning video files into searchable,
human-reviewable scene catalogs.

The first use case is film archive cataloging: a curator uploads or selects a
digitized film, the system detects scenes, extracts keyframes, generates visual
metadata, creates natural-language descriptions, embeds scenes for semantic
search, and lets humans correct the results.

The broader thesis is that many organizations have private visual collections
that are expensive to search because their metadata is incomplete. Archives,
media teams, insurers, inspection teams, and public institutions all face
versions of the same problem: the content is visual, the search need is textual,
and the data often cannot be sent to external APIs.

## Problem

Film archives often hold large collections where the available metadata is
sparse: title, date, duration, creator, and sometimes a short description. That
is not enough for day-to-day research questions such as:

- Which scenes show rural exteriors?
- Where do two people talk indoors?
- Which parts of a film contain vehicles, crowds, documents, or buildings?
- Where are the visually similar moments across a long work?

Manual cataloging is high-skill work and does not scale to every scene in every
film. The goal is not to replace curators. The goal is to give them a useful
first pass: searchable machine-generated metadata that can be reviewed,
corrected, exported, and improved over time.

## Product thesis

Public positioning:

> KUAA is an offline multimodal AI workbench that turns private video
> collections into searchable, human-reviewable metadata catalogs. The film
> archive is the first domain pack.

The product should demonstrate applied AI skill in a realistic setting:

- computer vision over real archival footage,
- local model execution,
- semantic image/text search,
- human-in-the-loop correction,
- reproducible artifacts,
- privacy-aware deployment,
- testable web application behavior.

## Current capabilities

Implemented today:

- Video inspection and frame extraction with FFmpeg/FFprobe.
- Scene detection and representative keyframe extraction with PySceneDetect.
- Visual analysis using face detection, object detection, and environment
  heuristics.
- CLIP embeddings for semantic text search and image search.
- Local Moondream 2 scene descriptions through configurable backends.
- Manual annotation of scene tags.
- FastAPI + HTMX web interface with Search, Scenes, Annotate, and Processing
  tabs.
- Server-sent event updates for processing jobs.
- Config-driven model backend registry using typed Protocols.
- Offline-oriented UI assets: local CSS, JavaScript, fonts, and icons.
- Regression tests for web routes, services, search, processing, i18n,
  accessibility, and pipeline behavior.

## Intended users

Primary user:

- A film archivist or curator who needs a searchable first-pass catalog for
  digitized audiovisual material.

Secondary users:

- Researchers looking for visual moments inside a film or collection.
- Media teams searching b-roll or historical footage.
- Technical reviewers evaluating an applied AI portfolio project.

Future adjacent users after domain packs:

- Broadcast/media asset teams.
- Insurance or inspection reviewers.
- Public-sector record managers.
- Industrial safety analysts working with private visual evidence.

## Why offline matters

Many useful visual collections cannot be casually uploaded to external AI APIs:

- rights may be unclear,
- material may be culturally sensitive,
- collections may be internal or private,
- institutions may have strict data-handling policies,
- archival workflows may happen on isolated machines.

This project treats local execution as a product constraint, not an afterthought.
Model weights may be downloaded on first setup, but video files, keyframes,
metadata, embeddings, annotations, and search queries are intended to stay on the
local machine.

## What this project is not

This is not a SaaS product. It currently has no accounts, billing, hosted
storage, multi-tenant permissions, or cloud processing pipeline.

This is not a replacement for archival judgment. Machine descriptions can be
wrong, incomplete, or culturally naive. The human annotation layer is part of
the product because the correct workflow is review and correction, not blind
automation.

This is not a legal compliance product. Offline execution can support privacy
and rights-sensitive workflows, but it does not by itself provide formal
certification, rights clearance, audit compliance, or policy enforcement.

## Portfolio value

For a career transition into applied ML or AI engineering, this project is meant
to show:

- domain insight from real archival work,
- ability to translate a messy institutional problem into a working system,
- model integration across computer vision and language,
- backend and frontend product execution,
- careful handling of failure cases and local constraints,
- evaluation mindset rather than demo-only thinking.

The strongest public story is not "I made an AI app for movies." It is:

> I used a real archive problem to build a local multimodal search system, then
> generalized it toward private visual-collection intelligence.

## Next proof points

The next phases of work should make the work easier to evaluate publicly:

- a reproducible public demo with non-private data,
- evaluation metrics for retrieval and metadata quality,
- domain pack configuration for at least one adjacent industry,
- run manifests and exports that make outputs traceable,
- a short case study and demo video for non-repo audiences.
