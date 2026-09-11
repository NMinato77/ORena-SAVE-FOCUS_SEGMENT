# ORena SAVE FOCUS — SEGMENT Track

このrepositoryは、[ORena SAVE FOCUS Challenge — SEGMENT Track](https://segment.orena-focus-challenge.org/)
に提出した`segment-algorithm`の公開用source snapshotです。

Docker submissionのentrypoint、推論処理、routing、format reducer、設定、
official sample fixtureを含みます。開発用の実験履歴・dataset・checkpointは
含めていません。

Qwen/SigLIPの大容量offline model assetsとfinal specialist weightsはGitHubへ
格納せず、`prepare_assets.sh`でbuild前にローカルから用意する構成です。
