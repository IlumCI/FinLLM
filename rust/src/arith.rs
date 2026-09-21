//! Exact integer-arithmetic evaluator and verifier.
//!
//! This is the *ground-truth oracle* for LAMb's self-play loop: every task the
//! proposer emits is evaluated here, and every answer the solver produces is
//! checked here. Because the reward signal for self-improvement is derived
//! entirely from this module, it must be exact (no floating point) and must
//! never panic on adversarial input -- hence a hand-written recursive-descent
//! parser over `i128` with checked arithmetic rather than any `eval`-like path.
//!
//! Grammar (standard precedence, left-associative):
//!   expr   := term   (('+' | '-') term)*
//!   term   := factor ('*' factor)*
//!   factor := ('+' | '-')* primary
//!   primary:= NUMBER | '(' expr ')'

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Num(i128),
    Plus,
    Minus,
    Star,
    LParen,
    RParen,
}

fn lex(s: &str) -> Option<Vec<Tok>> {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0usize;
    while i < bytes.len() {
        match bytes[i] {
            b' ' | b'\t' | b'\n' | b'\r' => i += 1,
            b'+' => {
                out.push(Tok::Plus);
                i += 1;
            }
            b'-' => {
                out.push(Tok::Minus);
                i += 1;
            }
            b'*' => {
                out.push(Tok::Star);
                i += 1;
            }
            b'(' => {
                out.push(Tok::LParen);
                i += 1;
            }
            b')' => {
                out.push(Tok::RParen);
                i += 1;
            }
            b'0'..=b'9' => {
                let start = i;
                while i < bytes.len() && bytes[i].is_ascii_digit() {
                    i += 1;
                }
                let num = std::str::from_utf8(&bytes[start..i]).ok()?;
                out.push(Tok::Num(num.parse::<i128>().ok()?));
            }
            _ => return None,
        }
    }
    Some(out)
}

struct Parser {
    tokens: Vec<Tok>,
    pos: usize,
}

impl Parser {
    fn peek(&self) -> Option<&Tok> {
        self.tokens.get(self.pos)
    }

    fn bump(&mut self) -> Option<Tok> {
        let t = self.tokens.get(self.pos).cloned();
        if t.is_some() {
            self.pos += 1;
        }
        t
    }

    fn parse_expr(&mut self) -> Option<i128> {
        let mut acc = self.parse_term()?;
        while let Some(t) = self.peek() {
            match t {
                Tok::Plus => {
                    self.pos += 1;
                    acc = acc.checked_add(self.parse_term()?)?;
                }
                Tok::Minus => {
                    self.pos += 1;
                    acc = acc.checked_sub(self.parse_term()?)?;
                }
                _ => break,
            }
        }
        Some(acc)
    }

    fn parse_term(&mut self) -> Option<i128> {
        let mut acc = self.parse_factor()?;
        while let Some(Tok::Star) = self.peek() {
            self.pos += 1;
            acc = acc.checked_mul(self.parse_factor()?)?;
        }
        Some(acc)
    }

    fn parse_factor(&mut self) -> Option<i128> {
        match self.peek() {
            Some(Tok::Minus) => {
                self.pos += 1;
                self.parse_factor()?.checked_neg()
            }
            Some(Tok::Plus) => {
                self.pos += 1;
                self.parse_factor()
            }
            _ => self.parse_primary(),
        }
    }

    fn parse_primary(&mut self) -> Option<i128> {
        match self.bump()? {
            Tok::Num(n) => Some(n),
            Tok::LParen => {
                let v = self.parse_expr()?;
                match self.bump()? {
                    Tok::RParen => Some(v),
                    _ => None,
                }
            }
            _ => None,
        }
    }
}

/// Evaluate an integer arithmetic expression exactly.
/// Returns `None` for malformed input or on overflow of `i128`.
pub fn evaluate(expr: &str) -> Option<i128> {
    let tokens = lex(expr)?;
    if tokens.is_empty() {
        return None;
    }
    let mut p = Parser { tokens, pos: 0 };
    let v = p.parse_expr()?;
    if p.pos != p.tokens.len() {
        return None; // trailing garbage
    }
    Some(v)
}

/// Verify that `answer` (a decimal integer string) equals `eval(expr)`.
pub fn verify(expr: &str, answer: &str) -> bool {
    match (evaluate(expr), answer.trim().parse::<i128>()) {
        (Some(v), Ok(a)) => v == a,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_ops() {
        assert_eq!(evaluate("2+3"), Some(5));
        assert_eq!(evaluate("10-4"), Some(6));
        assert_eq!(evaluate("6*7"), Some(42));
        assert_eq!(evaluate("2+3*4"), Some(14));
        assert_eq!(evaluate("(2+3)*4"), Some(20));
        assert_eq!(evaluate("-5+2"), Some(-3));
        assert_eq!(evaluate("7- -3"), Some(10));
    }

    #[test]
    fn rejects_garbage() {
        assert_eq!(evaluate(""), None);
        assert_eq!(evaluate("2+"), None);
        assert_eq!(evaluate("2 2"), None);
        assert_eq!(evaluate("2/3"), None); // division intentionally unsupported
        assert_eq!(evaluate("(2+3"), None);
    }

    #[test]
    fn verify_answers() {
        assert!(verify("123+877", "1000"));
        assert!(!verify("123+877", "999"));
        assert!(verify("-4*-4", "16"));
    }
}
