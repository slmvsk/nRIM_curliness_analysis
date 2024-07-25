#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 14:09:07 2024

@author: tetianasalamovska
"""

from cellprofiler_core.pipeline import Pipeline

pipeline = Pipeline()

pipeline.load("output/segment.cppipe")

measurements = pipeline.run()




