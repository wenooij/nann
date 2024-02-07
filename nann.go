package nann

import (
	"math"
	"math/rand"
	"slices"
	"strconv"
)

type Layer interface {
	Shape() (int, int)
	InitWeights(r *rand.Rand)
	Forward(in, out []float32) []float32
}

// DenseLayer represents dense weight matrix for a NANN layer.
//
// The fanIn is represented in the first dimension while the fanOut is the second dimension.
type DenseLayer [][]float32

func NewDenseLayer(fanIn, fanOut int) DenseLayer {
	d := make(DenseLayer, fanIn)
	for i := range d {
		d[i] = make([]float32, fanOut)
	}
	return d
}

func (l DenseLayer) Shape() (int, int) { return len(l), len(l[0]) }

func (l DenseLayer) InitWeights(r *rand.Rand) {
	fanIn := len(l)
	for _, row := range l {
		fanOut := len(row)
		for i := range row {
			row[i] = (2*r.Float32() - 1) * xavier(fanIn, fanOut)
		}
	}
}

func (l DenseLayer) Forward(in, out []float32) []float32 {
	if len(in) != len(l) {
		panic("shape mismatch (" + strconv.FormatInt(int64(len(in)), 10) + ", _)")
	}
	if len(out) != len(l[0]) {
		panic("shape mismatch (_, " + strconv.FormatInt(int64(len(out)), 10) + ")")
	}
	for i, x := range in {
		for j := range out {
			out[j] += l[i][j] * x
		}
	}
	return out
}

type Model struct {
	Layers  []Layer
	Biases  []float32
	ActivFn []ActivFn
}

func (m *Model) AddLayer(l Layer, activFn ActivFn) {
	m.Layers = append(m.Layers, l)
	m.Biases = append(m.Biases, 0)
	m.ActivFn = append(m.ActivFn, activFn)
}

func (m Model) Shape() (int, int) {
	s0, _ := m.Layers[0].Shape()
	_, s1 := m.Layers[len(m.Layers)-1].Shape()
	return s0, s1
}

func (m Model) InitWeights(r *rand.Rand) {
	for i := range m.Biases {
		m.Biases[i] = float32(r.NormFloat64()) / 64
	}
	for _, l := range m.Layers {
		l.InitWeights(r)
	}
}

func (m Model) Forward(in, out []float32) []float32 {
	for i := 0; ; {
		l := m.Layers[i]
		_, fanOut := l.Shape()
		if n := fanOut; n < len(out) {
			out = out[:n]
		} else if n > len(out) {
			out = slices.Grow(out, n-len(out))[:n]
		}
		out = l.Forward(in, out)
		b := m.Biases[i]
		a := m.ActivFn[i]
		for i, x := range out {
			out[i] = a.Apply(x + b)
		}
		if i++; i < len(m.Layers) {
			in, out = out, in
		} else {
			return out
		}
	}
}

type ActivFn interface {
	Apply(float32) float32
	Deriv(float32) float32
}

type activFn struct{ apply, deriv func(float32) float32 }

func (a activFn) Apply(x float32) float32 { return a.apply(x) }
func (a activFn) Deriv(x float32) float32 { return a.deriv(x) }

func Ident() ActivFn {
	return activFn{
		func(x float32) float32 { return x },
		func(float32) float32 { return 1 },
	}
}

func LRelu() ActivFn {
	return activFn{
		func(x float32) float32 {
			if x < 0 {
				return x / 64
			}
			return x
		},
		func(x float32) float32 {
			if x < 0 {
				return float32(1) / 64
			}
			return 1
		},
	}
}

func Sigmoid() ActivFn {
	return activFn{
		func(x float32) float32 { return 1 / (1 + exp(-x)) },
		func(x float32) float32 { return x * (1 - x) },
	}
}

func sum(xs []float32) float32 {
	var sum float32
	for _, x := range xs {
		sum += x
	}
	return sum
}

func softmaxd(xs []float32) float32 {
	var sp float32
	for _, x := range xs {
		sp += exp(x)
	}
	return sp
}

func Softmax(xs []float32) {
	sp := softmaxd(xs)
	for i, x := range xs {
		xs[i] = exp(x) / sp
	}
}

func exp(x float32) float32 { return float32(math.Exp(float64(x))) }

func sqrt(x float32) float32 { return float32(math.Sqrt(float64(x))) }

func xavier(in, out int) float32 { return sqrt(6 / (float32(in) + float32(out))) }
