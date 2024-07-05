#!/bin/bash
for ver in {1..30}
do
    python -m baseline.trigger_set.overwriting_attack --ver $ver
done