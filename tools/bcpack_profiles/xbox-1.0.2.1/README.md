# Xbox 1.0.2.1 calibration provenance

`calibration-v1.json` combines selector tables and confirmed predictor history from the reference decoder with provisional predictors from a local calibration profile. The BC7 shared seeds were converted into the Rust decoder's history order. The file contains a mixture of source-derived tables and empirically fitted values.

`entry-fits.json` collects 163 asset-key recipes assembled for the local catalog. They select strict decoding, supply endpoint histories, or apply BC7 field corrections at specific chunk offsets. For example, two BC4 endpoint pairs were chosen by comparing a fitted top mip with an independently stored lower mip and checking error on held-out blocks. These are numerical reconstruction parameters tied to asset identifiers. A matching pair of files can be passed through `--profile-dir`.

The two files are present in the draft for review. Redistribution of the complete mixed profile and catalog-derived recipes remains unresolved. The upstream source license alone does not establish the status of the locally fitted values. Public regression tests create their own predictors and asset identifiers. [Packaging discussion](https://github.com/R0Y00/Xbox2Windows/pull/836#issuecomment-7106298993)
