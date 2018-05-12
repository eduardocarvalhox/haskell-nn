import Numeric.LinearAlgebra
import Numeric.LinearAlgebra.Devel as V
import System.Random.MWC as Random

data PreLayer a = PreL (Vector a) (Matrix a) deriving Show
data Layer a = Lay (Vector a) (PreLayer a)

sigmoid :: R -> R
sigmoid x = 1 / (1 + (exp (-x)))

mapVector f = mapVectorWithIndex (\ i x -> f x)

forward :: Vector R -> [PreLayer R] -> ([Layer R], Vector R)
forward v [] = ([], v)
forward v ((PreL b w):xs) = let v' = mapVector sigmoid (flatten (w <> (asColumn v) + (asColumn b))) in
                            let (r, o) = forward v' xs in
                            ((Lay v (PreL b w)):r, o)

-- random generators
randomMat m n gen = (mapMatrixWithIndexM (\ _ _ -> uniformR (-1, 1) gen) (konst (1.0 :: Double) (m, n))) :: IO (Matrix R)
randomVec n gen = (mapVectorWithIndexM (\ _ _ -> uniformR (-1, 1) gen) (konst 1 n :: Vector Double)) :: IO (Vector R)

-- create [PreLayer R] from list of layers
-- input: [input_dim, hidden_layers_dim, output_dim]
-- setupNetwork :: [Int] -> [PreLayer R]
setupNetwork [a] _ = return []
setupNetwork (x:xs) gen = do
  let y = head xs
  b <- (randomVec y gen)
  m <- (randomMat y x gen)
  list <- (setupNetwork xs gen)
  return $ (PreL b m):list

main :: IO ()
main = do
  gen <- create
  input <- randomVec 10 gen
  layers <- setupNetwork [10, 100, 2] gen
  print $ snd(forward input layers)
