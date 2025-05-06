## Quick Start
... DSL snippet
-
-```glue
-model researcher {
-    adhesives = [glue, velcro]
-}
-```
+```glue
+model researcher {
+    provider = openrouter
+    adhesives = [glue, velcro]
+    config {
+        model = "meta-llama/llama-4-maverick:free"
+        planning_interval = 3  // Smolagents extra planning
+        system_prompt     = "You are an expert researcher who summarizes findings."
+    }
+}
+``` 