import Numeric.LinearAlgebra
import Numeric.LinearAlgebra.Devel as V
    
data PreLayer a = PreL (Vector a) (Matrix a)  
data Layer a = Lay (Vector a) (PreLayer a)

sigmoid :: R -> R
sigmoid x = 1 / (1 + (exp (-x)))

mapVector f = mapVectorWithIndex (\ i x -> f x)
            
forward :: Vector R -> [PreLayer R] -> ([Layer R], Vector R)
forward v [] = ([], v)
forward v ((PreL b w):xs) = let v' = mapVector sigmoid (flatten (w <> (asColumn v) + (asColumn b))) in
                            let (r, o) = forward v' xs in
                            ((Lay v (PreL b w)):r, o)
