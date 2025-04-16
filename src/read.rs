pub struct Reader {
    data: Vec<u8>,
    pos: usize,
}

impl Reader {
    fn from_file(path: &str) -> Self {
        let data = std::fs::read(path).expect("Failed to read file");
        Self {
            data,
            pos: 0,
        }
    }

    pub fn read_i32(&mut self) -> Result<i32, Error> {
        let mut buffer = [0; 4];
        buffer.copy_from_slice(self.data.get(self.pos..self.pos + 4)
            .ok_or(Error::InvalidPosition(self.pos))?
        );
        Ok(i32::from_le_bytes(buffer))
    }

    pub fn read_u32(&mut self) -> Result<u32, Error> {
        let mut buffer = [0; 4];
        buffer.copy_from_slice(self.data.get(self.pos..self.pos + 4)
            .ok_or(Error::InvalidPosition(self.pos))?
        );
        Ok(u32::from_le_bytes(buffer))
    }
}

pub enum Error {
    InvalidPosition(usize),
}
