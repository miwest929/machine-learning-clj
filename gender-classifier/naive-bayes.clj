(:require [clojure-csv.core :as csv])
(:require [clojure.java.io :as io]))

(defn load-dataset
  "Takes csv filename of the dataset as input and returns
   a List of the training examples"
   [fname]
   (with-open [file (io/reader fname)]
     (csv/parse-csv (slurp file))))
